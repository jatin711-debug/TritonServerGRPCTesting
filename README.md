# Triton YOLO11x Inference Server & Client

This repository contains a high-performance, GPU-accelerated implementation for deploying **YOLO11x** models via the **NVIDIA Triton Inference Server** using a TensorRT execution backend and a gRPC Python client.

By explicitly coupling the Triton Server with an ONNX-Runtime fallback mapped to an FP16 TensorRT Execution Accelerator, it achieves native <40ms raw compute latency (or ~190ms end-to-end on mobile 30W GPUs), bypassing expensive Python serialization bottlenecks.

## Directory Structure
- `setup_triton_yolo.py`: Automates dependency installation, exports YOLO11x PyTorch models to ONNX format, and scaffolds the Triton configuration (`config.pbtxt`).
- `triton_yolo_client.py`: A highly optimized Python client utilizing `tritonclient.grpc` to interact with the Triton Server natively. Includes modules for bounding box scaling offset padding and image batching.
- `model_repository/`: The directory where the Triton server detects the configured YOLO model and executes inference logic.

## Prerequisites
- NVIDIA GPU with CUDA Toolkit Installed
- Docker & NVIDIA Container Toolkit (for WSL2 / Linux)
- Python 3.10+

## 1. Setup & Starting Triton Server
Before running inference, you need to spin up the Triton Inference Server Docker container.

```bash
# Optional Setup if not already done manually
python setup_triton_yolo.py
```

Run the Triton Server explicitly mapping your local `model_repository` folder into the container to enable the dynamic TensorRT engine building:
```bash
docker run --gpus all --rm -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "d:\TritonTest\model_repository:/models" nvcr.io/nvidia/tritonserver:26.03-py3 tritonserver --model-repository=/models --strict-model-config=false
```
*Note: Depending on your hardware, the very first connection may hang for 2 to 5 minutes as Triton natively generates the FP16 `.plan` engine specific to your Linux OS / Docker container.*

## 2. Using the Client
Once the Triton Server has loaded up the backend metrics, run inference using the optimized gRPC client.

### Install Dependencies
```bash
pip install tritonclient[all] opencv-python numpy
```

### Infer on an Image
```bash
python .\triton_yolo_client.py --image .\img.png
```

### Infer on a Webcam/Video Stream
```bash
python .\triton_yolo_client.py --stream --device 0
```

## Performance Note
Ensure your device is plugged in if testing on mobile GPUs! Operating systems usually restrict low-power GPUs heavily while on battery power, increasing inference lag by up to 2x. 
