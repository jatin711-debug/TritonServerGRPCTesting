import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        """Initialize model parameters."""
        self.target_h = 640
        self.target_w = 640

    def preprocess_single(self, image):
        """Preprocess one image (HWC uint8 → CHW float32 normalized)."""

        h, w = image.shape[:2]

        # Scale factor
        scale = min(self.target_w / w, self.target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding
        canvas = np.full((self.target_h, self.target_w, 3), 114, dtype=np.uint8)

        top = (self.target_h - new_h) // 2
        left = (self.target_w - new_w) // 2

        canvas[top:top + new_h, left:left + new_w] = resized

        # Convert HWC → CHW
        canvas = canvas.transpose(2, 0, 1)

        # Normalize
        canvas = canvas.astype(np.float32) / 255.0

        return canvas

    def execute(self, requests):
        """Process incoming requests."""
        responses = []

        for request in requests:

            input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            input_data = input_tensor.as_numpy()  # shape: [B, H, W, 3]

            # Validate input
            if input_data.ndim != 4:
                raise ValueError(f"Expected 4D input [B,H,W,3], got {input_data.shape}")

            batch_size = input_data.shape[0]
            output_batch = []

            for i in range(batch_size):
                img = input_data[i]

                # Ensure uint8
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                processed = self.preprocess_single(img)
                output_batch.append(processed)

            # Stack batch → [B, 3, 640, 640]
            output_array = np.stack(output_batch).astype(np.float32)

            # Ensure contiguous memory (important for Triton)
            output_array = np.ascontiguousarray(output_array)

            output_tensor = pb_utils.Tensor("images", output_array)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses