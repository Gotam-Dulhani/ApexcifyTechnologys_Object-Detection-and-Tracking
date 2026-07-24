#!/bin/bash
set -e

echo "=== Installing ultralytics for ONNX export ==="
pip install ultralytics

echo "=== Exporting yolov8n.pt to ONNX ==="
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True)"

echo "=== Verifying ONNX model ==="
if [ ! -f yolov8n.onnx ]; then
  echo "ERROR: yolov8n.onnx was not created"
  exit 1
fi
ls -lh yolov8n.onnx

echo "=== Installing runtime dependencies ==="
pip install -r requirements.txt

echo "=== Removing heavy build-only packages ==="
pip uninstall -y ultralytics torch torchvision

echo "=== Purging pip cache ==="
pip cache purge 2>/dev/null || true

echo "=== Build complete ==="
