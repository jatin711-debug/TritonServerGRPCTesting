"""
Triton Inference Server setup for YOLO11x with TensorRT engine.
This script:
1. Exports YOLO11x to ONNX format
2. Builds TensorRT engine from ONNX
3. Sets up Triton model repository
4. Starts Triton server (or generates docker run command)
"""

import subprocess
import os
import sys
from pathlib import Path

# Configuration
MODEL_NAME = "yolo11x"
MODEL_VERSION = "1"
WORKSPACE = Path("d:/TritonTest")
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
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
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
    print("\n[2/5] Exporting YOLO11x to ONNX...")

    onnx_path = MODEL_REPO / f"{MODEL_NAME}.onnx"
    MODEL_REPO.mkdir(parents=True, exist_ok=True)

    # Export using ultralytics
    export_script = f"""
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.export(format='onnx', opset=12, simplify=True)
"""
    # For simplicity, we'll use trtexec to convert directly from PyTorch
    # First download/load the model
    cmd = [
        sys.executable, "-c",
        f"""
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.export(format='onnx', opset=12, simplify=True)
print('ONNX export complete')
"""
    ]
    run_command(cmd, "Exporting YOLO11x to ONNX")
    return onnx_path

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
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
    reshape {{ shape: [1, 3, 640, 640] }}
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [1, 84, 8400]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator {{
      name: "tensorrt"
      parameters {{
        key: "precision_mode"
        value: "FP16"
      }}
      parameters {{
        key: "max_workspace_size_bytes"
        value: "10737418240"
      }}
    }}
  }}
}}
"""

    config_path = TRITON_MODEL_REPO / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config)

    print(f"Config saved to: {config_path}")
    return config_path

def copy_engine_to_model_repo(engine_path):
    """Copy TensorRT engine to Triton model repository."""
    print("\n[5/5] Copying engine to model repository...")
    dest = TRITON_MODEL_REPO / "model.plan"
    import shutil
    if engine_path.exists():
        shutil.copy(engine_path, dest)
        print(f"Engine copied to: {dest}")
    else:
        print(f"WARNING: Engine not found at {engine_path}")
        print("You may need to build the engine manually or fix the ONNX export path")

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

    # Step 3: Build TensorRT engine
    engine_path = build_tensorrt_engine(onnx_path)

    # Step 4: Create Triton config
    create_triton_config()

    # Step 5: Copy engine to model repo
    copy_engine_to_model_repo(engine_path)

    # Generate Docker command
    generate_docker_command()

    print("\nSetup complete! Run the docker command above to start Triton server.")