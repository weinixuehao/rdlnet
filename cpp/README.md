## RDLNet TFLite C++ inference (demo)

This folder contains a minimal C++ program that runs inference on the exported `rdlnet_points.tflite` / `rdlnet_full.tflite`.

### 1) Export the TFLite model

From repo root:

```bash
./.venv/bin/python scripts/export_rdlnet_tflite.py \
  --ckpt /path/to/rdlnet_best.pt \
  --out-dir ./output/_export_rdlnet \
  --img-size 1024 \
  --export points \
  --input-range 0_1 \
  --fp16
```

The export script always applies SAM mean/std normalization, so **SAM mean/std normalization is already inside the model graph**.
That means the C++ side should just feed float32 RGB pixels in the chosen `--input-range` (`0_1` or `0_255`).

`--fp16` uses float16 weight quantization. Typically **I/O stays float32**, but the demo also supports float16 I/O.

### 2) Build the C++ demo

You need:
- TensorFlow Lite headers + library
- OpenCV (for image loading / resize)

Configure and build:

```bash
cmake -S cpp -B cpp/build \
  -DTFLITE_INCLUDE_DIR=/path/to/tensorflow \
  -DTFLITE_LIBRARY=/path/to/libtensorflowlite.a
cmake --build cpp/build -j
```

### 3) Run

With OpenCV (`--image`):

```bash
./cpp/build/rdlnet_tflite_infer \
  --model ./output/_export_rdlnet/rdlnet_points.tflite \
  --image /path/to/image.jpg \
  --input-range 0_1 \
  --doc-class-id 0
```

Output:
- prints `pred_logits` / `pred_points` tensor shapes
- selects query \(q^\*\) by the highest `softmax(pred_logits[q])[doc_class_id]`
- prints the selected `pred_points[q*]` in normalized \([0,1]\) and pixel coordinates

