"""
Export YOLO11s ONNX with batch-only dynamic dimensions.
Input:  [-1, 3, 640, 640]  (batch dynamic, spatial fixed)
Output: [-1, 84, 8400]     (batch dynamic, detections fixed)
"""
from ultralytics import YOLO
import onnx
import shutil
from pathlib import Path

MODEL_NAME = "yolo11s"
DEST = Path("model_repository/yolo11s/1/model.onnx")

# Step 1: Export with dynamic=True (makes everything dynamic)
print("Step 1: Exporting ONNX with dynamic=True ...")
model = YOLO(f"{MODEL_NAME}.pt")
export_path = model.export(format="onnx", opset=12, simplify=True, dynamic=True)

# Step 2: Fix spatial dims, keep only batch dynamic
print("Step 2: Fixing spatial dims to 640x640 (batch-only dynamic) ...")
onnx_model = onnx.load(export_path)

for inp in onnx_model.graph.input:
    shape = inp.type.tensor_type.shape
    shape.dim[0].dim_param = "batch"
    shape.dim[0].ClearField("dim_value")
    shape.dim[1].dim_value = 3
    shape.dim[1].ClearField("dim_param")
    shape.dim[2].dim_value = 640
    shape.dim[2].ClearField("dim_param")
    shape.dim[3].dim_value = 640
    shape.dim[3].ClearField("dim_param")

for out in onnx_model.graph.output:
    shape = out.type.tensor_type.shape
    shape.dim[0].dim_param = "batch"
    shape.dim[0].ClearField("dim_value")
    shape.dim[1].dim_value = 84
    shape.dim[1].ClearField("dim_param")
    shape.dim[2].dim_value = 8400
    shape.dim[2].ClearField("dim_param")

onnx.save(onnx_model, export_path)

# Verify
print("\nVerification:")
m = onnx.load(export_path)
for inp in m.graph.input:
    dims = [d.dim_param or str(d.dim_value) for d in inp.type.tensor_type.shape.dim]
    print(f"  Input  '{inp.name}': [{', '.join(dims)}]")
for out in m.graph.output:
    dims = [d.dim_param or str(d.dim_value) for d in out.type.tensor_type.shape.dim]
    print(f"  Output '{out.name}': [{', '.join(dims)}]")

# Step 3: Copy to model repo
DEST.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(export_path, DEST)
print(f"\nCopied to: {DEST}")
print("Done!")
