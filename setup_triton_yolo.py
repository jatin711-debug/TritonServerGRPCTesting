"""
Triton Inference Server setup for YOLO11x with TensorRT engine.
This script:
1. Exports YOLO11x to ONNX format
2. Builds TensorRT engine from ONNX
3. Sets up Triton model repository
4. Starts Triton server (or generates docker run command)
"""

import shutil
import subprocess
import os
import sys
from pathlib import Path

# Configuration
MODEL_NAME = "yolo11s"
MODEL_VERSION = "1"
WORKSPACE = Path(__file__).parent.resolve()
MODEL_REPO = WORKSPACE / "model_repository"
TRITON_MODEL_REPO = MODEL_REPO / MODEL_NAME / MODEL_VERSION

# YOLO11x parameters
INPUT_SHAPE = (1, 3, 640, 640)  # batch, channels, height, width
IMAGENET_CLASSES = 80  # COCO dataset

def run_command(cmd, description):
    """Run shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    # Convert list to string for better Windows shell handling
    cmd_str = " ".join([f'"{c}"' if " " in str(c) else str(c) for c in cmd])
    print(f"Command: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd=WORKSPACE)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True

def install_dependencies():
    """Install required packages."""
    print("\n[1/5] Installing dependencies...")
    packages = [
        "ultralytics>=8.2.0",  # For YOLO11
        "torch",               # For model export
        "tensorrt",           # For TensorRT Python API
    ]
    for pkg in packages:
        run_command([sys.executable, "-m", "pip", "install", pkg, "-q"],
                   f"Installing {pkg}")

def export_to_onnx():
    """Export YOLO11x to ONNX format."""
    print("\n[2/5] Exporting YOLO11s to ONNX...")

    from ultralytics import YOLO

    onnx_dest = MODEL_REPO / f"{MODEL_NAME}.onnx"
    MODEL_REPO.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}.pt...")
    model = YOLO(f"{MODEL_NAME}.pt")
    print("Exporting to ONNX...")
    export_path = model.export(format="onnx", opset=12, simplify=True)
    print(f"Ultralytics exported to: {export_path}")

    if export_path and os.path.exists(export_path):
        if os.path.exists(onnx_dest):
            os.remove(onnx_dest)
        shutil.move(export_path, onnx_dest)
        print(f"SUCCESS: Moved to {onnx_dest}")
        return onnx_dest
    else:
        print("ERROR: Could not find exported ONNX file.")
        return None

def build_tensorrt_engine(onnx_path):
    """Build TensorRT engine from ONNX using trtexec."""
    print("\n[3/5] Building TensorRT engine...")

    engine_path = MODEL_REPO / f"{MODEL_NAME}.engine"

    # Format: images:1x3x640x640 (no parentheses, no spaces, use x as separator)
    shape_str = "x".join(str(d) for d in INPUT_SHAPE)  # "1x3x640x640"

    cmd = [
        "trtexec",
        "--onnx=" + str(onnx_path),
        "--fp16",                    # Half precision for speed
        "--useCudaGraph",            # Enable CUDA graphs
        f"--minShapes=images:{shape_str}",
        f"--optShapes=images:{shape_str}",
        f"--maxShapes=images:{shape_str}",
        "--saveEngine=" + str(engine_path),
    ]

    run_command(cmd, "Building TensorRT engine with trtexec")
    return engine_path

def create_triton_config():
    """Create Triton model config.pbtxt."""
    print("\n[4/5] Creating Triton model configuration...")

    TRITON_MODEL_REPO.mkdir(parents=True, exist_ok=True)

    config = f"""
name: "{MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 2
  }}
]

dynamic_batching {{
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}}

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator {{
      name: "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
    }}
  }}
}}
"""

    config_path = TRITON_MODEL_REPO / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config)

    print(f"Config saved to: {config_path}")
    return config_path

def copy_engine_to_model_repo(onnx_path):
    """Copy ONNX model to Triton model repository."""
    print("\n[5/5] Copying model to model repository...")
    dest = TRITON_MODEL_REPO / "model.onnx"
    if onnx_path.exists():
        shutil.copy(onnx_path, dest)
        print(f"Model copied to: {dest}")
    else:
        print(f"WARNING: ONNX model not found at {onnx_path}")

def generate_docker_command():
    """Generate the Docker command to run Triton server."""
    print("\n" + "="*60)
    print("TRITON DOCKER COMMAND")
    print("="*60)
    model_path = os.path.abspath(MODEL_REPO)
    docker_cmd = f"""
docker run --gpus all \\
    --rm \\
    -p 8000:8000 \\
    -p 8001:8001 \\
    -p 8002:8002 \\
    -v "{model_path}:/models" \\
    nvcr.io/nvidia/tritonserver:26.03-py3 \\
    tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1
"""
    print(docker_cmd)
    return docker_cmd

if __name__ == "__main__":
    print("Triton YOLO11x TensorRT Setup")
    print("="*60)

    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Export to ONNX
    onnx_path = export_to_onnx()

    # Step 3: Build TensorRT engine (skipped - using ONNX with onnxruntime)
    # engine_path = build_tensorrt_engine(onnx_path)

    # Step 4: Create Triton config
    create_triton_config()

    # Step 5: Copy ONNX model to model repo
    copy_engine_to_model_repo(onnx_path)

    # Generate Docker command
    generate_docker_command()

    print("\nSetup complete! Run the docker command above to start Triton server.")