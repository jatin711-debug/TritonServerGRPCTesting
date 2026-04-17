"""
Triton Inference Client for YOLO11x with TensorRT.
Sends images to Triton server and receives detections.

Usage:
    python triton_yolo_client.py --image path/to/image.jpg
    python triton_yolo_client.py --image path/to/image.jpg --stream  # for video stream
"""

import numpy as np
import cv2
import argparse
import time
from typing import List, Tuple, Optional
import tracemalloc
import tritonclient.grpc as grpcclient
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

# Triton server settings
TRITON_URL = "localhost:8001"
MODEL_NAME = "yolo11s"
INPUT_NAME = "images"
OUTPUT_NAME = "output0"

# COCO class names (YOLO11x default)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image for YOLO11x: letterbox resize + normalization."""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with letterbox
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    # Convert to CHW format and normalize
    canvas = canvas.transpose(2, 0, 1)  # HWC -> CHW
    canvas = canvas.astype(np.float32) / 255.0

    # Add batch dimension
    canvas = np.expand_dims(canvas, axis=0)
    canvas = np.ascontiguousarray(canvas)

    return canvas, scale, (top, left), (new_w, new_h)

def postprocess_detections(
    output: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """Postprocess YOLO output to get bounding boxes."""
    predictions = output[0]  # Remove batch dimension

    # Transpose: [84, 8400] -> [8400, 84]
    predictions = predictions.T

    # Split into box coords and class scores
    # First 4 columns are bbox (cx, cy, w, h), remaining are class scores
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # Get class IDs and scores
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert from cx,cy,w,h to x1,y1,x2,y2
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Scale boxes back to original image size
    boxes = np.column_stack([x1, y1, x2, y2])

    # Apply padding offset
    pad_top, pad_left = pad
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top

    # Scale to original image
    orig_h, orig_w = orig_shape
    boxes /= scale

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidences.tolist(),
        conf_threshold, iou_threshold
    )

    results = []
    if len(indices) > 0:
        for idx in indices:
            class_id = int(class_ids[idx])
            conf = float(confidences[idx])
            x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
            results.append((class_id, conf, (x1, y1, x2, y2)))

    return results

def infer_image_triton(image_path: str, visualize: bool = True, output_path: str = None) -> dict:
    """Send image to Triton server for inference."""
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = image.shape[:2]
    input_data, scale, pad, _ = preprocess_image(image)

    # Prepare request
    # Build inference request
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to create triton client: {e}")

    inputs = [grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    # Send inference request
    tracemalloc.start()
    start_time = time.perf_counter()

    result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    inference_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Parse output
    output_data = result.as_numpy(OUTPUT_NAME)

    # Postprocess
    detections = postprocess_detections(output_data, scale, pad, (orig_h, orig_w))

    # Visualize if requested
    if visualize:
        for class_id, conf, (x1, y1, x2, y2) in detections:
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save to file instead of imshow (works in headless environments)
        if output_path is None:
            output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
        cv2.imwrite(output_path, image)
        print(f"Result saved to: {output_path}")

    return {
        "detections": detections,
        "inference_time_ms": inference_time * 1000,
        "peak_memory_mb": peak / (1024 * 1024)
    }


def infer_image_concurrent(image_path: str, request_id: int, output_dir: str = "outputs", triton_client=None) -> dict:
    """Single inference request for concurrent execution."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"result_{request_id:04d}.jpg")

    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = image.shape[:2]
    input_data, scale, pad, _ = preprocess_image(image)

    if triton_client is None:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)

    inputs = [grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    start_time = time.perf_counter()
    result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    inference_time = time.perf_counter() - start_time

    output_data = result.as_numpy(OUTPUT_NAME)
    if output_data is None:
        raise RuntimeError(f"Request #{request_id}: Got None output from server")

    detections = postprocess_detections(output_data, scale, pad, (orig_h, orig_w))

    # Visualize
    for class_id, conf, (x1, y1, x2, y2) in detections:
        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add request ID overlay
    cv2.putText(image, f"Request #{request_id}", (10, orig_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imwrite(output_path, image)

    return {
        "request_id": request_id,
        "output_path": output_path,
        "detections": len(detections),
        "inference_time_ms": inference_time * 1000
    }


def run_concurrent_inference(image_path: str, num_requests: int = 50, max_workers: int = 10):
    """Run n concurrent requests to Triton server."""
    print(f"Starting {num_requests} concurrent requests to {TRITON_URL}")
    print(f"Model: {MODEL_NAME}, Image: {image_path}")
    print(f"Max workers: {max_workers} (limited to avoid connection overload)")
    print("-" * 50)

    # Create ONE client and reuse it (thread-safe for grpc client)
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print("Triton client connected successfully")

    results = []
    errors = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(infer_image_concurrent, image_path, i, "outputs", triton_client): i
            for i in range(num_requests)
        }

        for future in as_completed(futures):
            req_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Request #{req_id:04d} completed in {result['inference_time_ms']:.2f}ms")
            except Exception as e:
                errors.append((req_id, str(e)))
                print(f"Request #{req_id:04d} FAILED: {e}")

    total_time = time.perf_counter() - start_time

    print("-" * 50)
    print(f"Completed: {len(results)}/{num_requests} successful, {len(errors)} failed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(results)/total_time:.2f} req/s")
    print(f"Avg latency: {sum(r['inference_time_ms'] for r in results)/len(results):.2f}ms" if results else "N/A")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for req_id, err in errors[:5]:
            print(f"  Request #{req_id}: {err}")

    return results, errors

def infer_video_stream(video_path: int = 0):
    """Run inference on video stream."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")

    print(f"Streaming from device {video_path}... Press 'q' to quit")

    frame_count = 0
    total_inference_time = 0

    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to create triton client: {e}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        input_data, scale, pad, _ = preprocess_image(frame)

        # Build request
        inputs = [grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        # Infer
        start = time.perf_counter()
        result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        inference_time = (time.perf_counter() - start) * 1000
        total_inference_time += inference_time
        frame_count += 1

        output_data = result.as_numpy(OUTPUT_NAME)
        if output_data is not None:
            detections = postprocess_detections(output_data, scale, pad, (orig_h, orig_w))

            # Draw
            for class_id, conf, (x1, y1, x2, y2) in detections:
                label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS overlay
        avg_time = total_inference_time / frame_count if frame_count > 0 else 0
        cv2.putText(frame, f"FPS: {1000/avg_time:.1f} | Latency: {avg_time:.1f}ms",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save frame instead of imshow in headless mode
        cv2.imwrite(f"frame_{frame_count:04d}.jpg", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames, avg latency: {total_inference_time/frame_count:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton YOLO11x Inference Client")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--stream", action="store_true", help="Use video stream (default: webcam)")
    parser.add_argument("--device", type=int, default=0, help="Video device index (default: 0)")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton server URL")
    parser.add_argument("--concurrent", type=int, default=0, help="Run N concurrent requests (default: 0 = single inference)")
    parser.add_argument("--workers", type=int, default=50, help="Max concurrent workers (default: 50)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for concurrent results")

    args = parser.parse_args()
    TRITON_URL = args.url

    if args.concurrent > 0:
        if not args.image:
            parser.error("--concurrent requires --image")
        run_concurrent_inference(args.image, num_requests=args.concurrent, max_workers=args.workers)
    elif args.image:
        result = infer_image_triton(args.image, visualize=True)
        print(f"\nResults:")
        print(f"  Detections: {len(result['detections'])}")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"  Peak memory: {result['peak_memory_mb']:.2f} MB")
        for class_id, conf, bbox in result['detections'][:5]:  # Show first 5
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            print(f"  - {label}: {conf:.3f} at {bbox}")
    elif args.stream:
        infer_video_stream(args.device)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python triton_yolo_client.py --image photo.jpg")
        print("  python triton_yolo_client.py --stream --device 0")
        print("  python triton_yolo_client.py --image photo.jpg --concurrent 50")