"""
Build TensorRT engine for YOLO11x - simplified version.
Run this before starting Triton server.
"""
import subprocess
import os
from pathlib import Path

WORKSPACE = Path("d:/TritonTest")
MODEL_REPO = WORKSPACE / "model_repository"
ONNX_PATH = MODEL_REPO / "yolo11x.onnx"
ENGINE_PATH = MODEL_REPO / "yolo11x.engine"

print("=" * 60)
print("TensorRT Engine Builder for YOLO11x")
print("=" * 60)

# Check if ONNX exists
if not ONNX_PATH.exists():
    print(f"\nERROR: ONNX model not found at {ONNX_PATH}")
    print("Please run the ONNX export first:")
    print("  python -c \"from ultralytics import YOLO; YOLO('yolo11x.pt').export(format='onnx', opset=12)\"")
    exit(1)

print(f"\nONNX model: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

# Build engine with trtexec
# Using simplified command without deprecated flags
cmd = [
    "trtexec",
    "--onnx=" + str(ONNX_PATH),
    "--fp16",
    "--useCudaGraph",
    "--saveEngine=" + str(ENGINE_PATH),
    "--verbose",  # Show detailed build progress
]

print(f"\nBuilding TensorRT engine...")
print(f"Command: {' '.join(cmd)}")

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print(f"\nERROR: trtexec failed with return code {result.returncode}")
    print("\nTroubleshooting tips:")
    print("  1. Make sure CUDA and TensorRT are properly installed")
    print("  2. Try running trtexec directly to see detailed errors")
    exit(1)

# Copy engine to Triton model repository
TRITON_MODEL_PATH = MODEL_REPO / "yolo11x" / "1" / "model.plan"
import shutil
if ENGINE_PATH.exists():
    shutil.copy(ENGINE_PATH, TRITON_MODEL_PATH)
    print(f"\nEngine copied to: {TRITON_MODEL_PATH}")
    print(f"Engine size: {TRITON_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"\nWARNING: Engine not found at {ENGINE_PATH}")
    exit(1)

print("\n" + "=" * 60)
print("Engine build complete!")
print("=" * 60)
print("\nNow start Triton server with:")
print(f"  docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
print(f"    -v \"{MODEL_REPO}:/models\" \\")
print("    nvcr.io/nvidia/tritonserver:26.03-py3 \\")
print("    tritonserver --model-repository=/models --strict-model-config=false")