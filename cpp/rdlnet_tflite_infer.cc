#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct Args {
  std::string model_path;
  std::string image_path;
  std::string vis_out;
  int img_size = 1024;
  std::string input_range = "0_1";  // 0_1 | 0_255
  int doc_class_id = 0;
};

static void die(const std::string& msg) {
  std::cerr << "ERROR: " << msg << "\n";
  std::exit(2);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) die("Missing value for " + name);
      return std::string(argv[++i]);
    };
    if (k == "--model") {
      a.model_path = need("--model");
    } else if (k == "--image") {
      a.image_path = need("--image");
    } else if (k == "--vis-out") {
      a.vis_out = need("--vis-out");
    } else if (k == "--img-size") {
      a.img_size = std::stoi(need("--img-size"));
    } else if (k == "--input-range") {
      a.input_range = need("--input-range");
    } else if (k == "--doc-class-id") {
      a.doc_class_id = std::stoi(need("--doc-class-id"));
    } else if (k == "-h" || k == "--help") {
      std::cout
          << "Usage:\n"
          << "  rdlnet_tflite_infer --model rdlnet_points.tflite --image test.jpg \\\n"
          << "    --vis-out out.png [--img-size 1024] [--input-range 0_1|0_255] [--doc-class-id 0]\n\n"
          << "Notes:\n"
          << "  - If you exported with scripts/export_rdlnet_tflite.py, SAM mean/std normalization\n"
          << "    is already inside the model graph.\n"
          << "  - Export with --fp16 usually keeps I/O as float32; this demo also supports float16 I/O.\n";
      std::exit(0);
    } else {
      die("Unknown arg: " + k);
    }
  }
  if (a.model_path.empty()) die("Missing --model");
  if (a.image_path.empty()) die("Missing --image");
  if (a.vis_out.empty()) die("Missing --vis-out");
  if (!(a.input_range == "0_1" || a.input_range == "0_255")) die("--input-range must be 0_1 or 0_255");
  return a;
}

static std::vector<float> softmax(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
  std::vector<float> e(n);
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    e[i] = std::exp(x[i] - m);
    s += e[i];
  }
  if (s <= 0.0f) return std::vector<float>(n, 1.0f / std::max(1, n));
  for (int i = 0; i < n; ++i) e[i] /= s;
  return e;
}

// --- float16 helpers (IEEE 754 half) ---
static inline uint16_t float_to_half_bits(float f) {
  union {
    uint32_t u;
    float f;
  } v;
  v.f = f;
  uint32_t x = v.u;
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t mantissa = x & 0x007fffffu;
  int exp = static_cast<int>((x >> 23) & 0xffu) - 127;

  if (exp > 15) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // inf
  }
  if (exp <= -15) {
    if (exp < -24) return static_cast<uint16_t>(sign);  // underflow -> 0
    mantissa |= 0x00800000u;
    int shift = -exp - 1;
    uint32_t m = mantissa >> (shift + 13);
    // round to nearest
    if ((mantissa >> (shift + 12)) & 1u) m += 1;
    return static_cast<uint16_t>(sign | m);
  }

  uint16_t he = static_cast<uint16_t>(exp + 15);
  uint16_t hm = static_cast<uint16_t>(mantissa >> 13);
  // round to nearest
  if (mantissa & 0x00001000u) {
    hm += 1;
    if (hm == 0x0400u) {  // mantissa overflow
      hm = 0;
      he += 1;
      if (he >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (he << 10) | hm);
}

static inline float half_bits_to_float(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = static_cast<uint32_t>(h) & 0x03ffu;
  uint32_t out;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      // subnormal
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x03ffu;
      uint32_t e = (exp - 1 + 127) << 23;
      uint32_t m = mant << 13;
      out = sign | e | m;
    }
  } else if (exp == 31) {
    // inf/nan
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    uint32_t e = (exp - 15 + 127) << 23;
    uint32_t m = mant << 13;
    out = sign | e | m;
  }
  union {
    uint32_t u;
    float f;
  } v;
  v.u = out;
  return v.f;
}

struct InputLayout {
  // For input tensor rank-4.
  // NCHW: [1,3,H,W]
  // NHWC: [1,H,W,3]
  bool is_nchw = true;
  int n = 1;
  int c = 3;
  int h = 0;
  int w = 0;
};

static InputLayout infer_layout(const TfLiteTensor* t) {
  if (!t || !t->dims || t->dims->size != 4) die("Expected input tensor rank 4");
  const int d0 = t->dims->data[0];
  const int d1 = t->dims->data[1];
  const int d2 = t->dims->data[2];
  const int d3 = t->dims->data[3];

  InputLayout l;
  l.n = d0;
  if (d1 == 3) {  // NCHW
    l.is_nchw = true;
    l.c = d1;
    l.h = d2;
    l.w = d3;
    return l;
  }
  if (d3 == 3) {  // NHWC
    l.is_nchw = false;
    l.h = d1;
    l.w = d2;
    l.c = d3;
    return l;
  }
  die("Cannot infer input layout: expected channel dim to be 3 (NCHW or NHWC)");
  return l;
}

static std::vector<float> load_rgb_resized_hwc_f32_with_opencv(
    const std::string& path, int img_size, const std::string& input_range) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) die("Failed to read image: " + path);

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(img_size, img_size), 0, 0, cv::INTER_LINEAR);

  cv::Mat f32;
  resized.convertTo(f32, CV_32FC3);
  if (input_range == "0_1") {
    f32 *= (1.0f / 255.0f);
  }
  std::vector<float> out(static_cast<size_t>(img_size) * static_cast<size_t>(img_size) * 3u);
  std::memcpy(out.data(), f32.data, out.size() * sizeof(float));
  return out;  // HWC interleaved RGB float32
}

static void write_input_tensor_f32(float* dst, const InputLayout& l, const std::vector<float>& img_hwc_f32) {
  const size_t expected = static_cast<size_t>(l.h) * static_cast<size_t>(l.w) * 3u;
  if (img_hwc_f32.size() != expected) die("Input RGB buffer size mismatch");
  if (l.n != 1) die("This demo expects batch=1 (export uses fixed batch).");

  const int H = l.h;
  const int W = l.w;
  if (l.is_nchw) {
    // dst index: ((c*H)+y)*W + x
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t base = (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * 3u;
        dst[(0 * H + y) * W + x] = img_hwc_f32[base + 0];
        dst[(1 * H + y) * W + x] = img_hwc_f32[base + 1];
        dst[(2 * H + y) * W + x] = img_hwc_f32[base + 2];
      }
    }
  } else {
    // NHWC contiguous: ((y*W)+x)*3 + c
    std::memcpy(dst, img_hwc_f32.data(), img_hwc_f32.size() * sizeof(float));
  }
}

static void write_input_tensor_f16(uint16_t* dst, const InputLayout& l, const std::vector<float>& img_hwc_f32) {
  const size_t expected = static_cast<size_t>(l.h) * static_cast<size_t>(l.w) * 3u;
  if (img_hwc_f32.size() != expected) die("Input RGB buffer size mismatch");
  if (l.n != 1) die("This demo expects batch=1 (export uses fixed batch).");

  const int H = l.h;
  const int W = l.w;
  if (l.is_nchw) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t base = (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * 3u;
        dst[(0 * H + y) * W + x] = float_to_half_bits(img_hwc_f32[base + 0]);
        dst[(1 * H + y) * W + x] = float_to_half_bits(img_hwc_f32[base + 1]);
        dst[(2 * H + y) * W + x] = float_to_half_bits(img_hwc_f32[base + 2]);
      }
    }
  } else {
    for (size_t i = 0; i < img_hwc_f32.size(); ++i) dst[i] = float_to_half_bits(img_hwc_f32[i]);
  }
}

static std::string dims_to_string(const TfLiteTensor* t) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < t->dims->size; ++i) {
    if (i) oss << ",";
    oss << t->dims->data[i];
  }
  oss << "]";
  return oss.str();
}

static cv::Mat load_rgb_resized_u8_with_opencv(const std::string& path, int img_size) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) die("Failed to read image: " + path);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(img_size, img_size), 0, 0, cv::INTER_LINEAR);
  return resized;  // uint8 RGB
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  auto model = tflite::FlatBufferModel::BuildFromFile(args.model_path.c_str());
  if (!model) die("Failed to load tflite model: " + args.model_path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder(*model, resolver)(&interp);
  if (!interp) die("Failed to build tflite interpreter");

  if (interp->AllocateTensors() != kTfLiteOk) die("AllocateTensors failed");

  if (interp->inputs().size() != 1) die("Expected exactly 1 input tensor");
  const int in_idx = interp->inputs()[0];
  TfLiteTensor* in = interp->tensor(in_idx);
  if (!(in->type == kTfLiteFloat32 || in->type == kTfLiteFloat16)) die("Expected float32/float16 input tensor");

  InputLayout layout = infer_layout(in);
  const int H = layout.h;
  const int W = layout.w;
  if (H != args.img_size || W != args.img_size) {
    std::cerr << "WARN: model input is " << dims_to_string(in) << ", but --img-size is " << args.img_size
              << ". Using model tensor size.\n";
  }

  std::vector<float> img_hwc_f32;
  img_hwc_f32 = load_rgb_resized_hwc_f32_with_opencv(args.image_path, H, args.input_range);

  if (in->type == kTfLiteFloat32) {
    float* in_data = interp->typed_tensor<float>(in_idx);
    write_input_tensor_f32(in_data, layout, img_hwc_f32);
  } else {
    uint16_t* in_data = reinterpret_cast<uint16_t*>(interp->tensor(in_idx)->data.raw);
    if (!in_data) die("Null input tensor buffer");
    write_input_tensor_f16(in_data, layout, img_hwc_f32);
  }

  if (interp->Invoke() != kTfLiteOk) die("Invoke failed");

  // Export script returns:
  // - points mode: (pred_logits, pred_points)
  // - full mode:   (pred_logits, pred_masks, pred_points)
  auto outs = interp->outputs();
  if (!(outs.size() == 2 || outs.size() == 3)) {
    die("Unexpected output tensor count (expected 2 or 3), got " + std::to_string(outs.size()));
  }

  // TFLite does not guarantee output tensor ordering. In points mode (2 outputs),
  // `pred_points` has last dim 2P (even), while `pred_logits` has last dim C.
  if (outs.size() == 2) {
    auto lastdim = [&](int oi) -> int { return interp->tensor(oi)->dims->data[2]; };
    if (lastdim(outs[0]) % 2 == 0) std::swap(outs[0], outs[1]);
  }

  const TfLiteTensor* t0 = interp->tensor(outs[0]);
  const TfLiteTensor* t_last = interp->tensor(outs.back());
  if (!((t0->type == kTfLiteFloat32 || t0->type == kTfLiteFloat16) &&
        (t_last->type == kTfLiteFloat32 || t_last->type == kTfLiteFloat16))) {
    die("Expected float32/float16 outputs");
  }
  if (!t0->dims || t0->dims->size != 3) die("pred_logits should be rank-3: [B,Nq,C]");
  if (!t_last->dims || t_last->dims->size != 3) die("pred_points should be rank-3: [B,Nq,2P]");

  const int B = t0->dims->data[0];
  const int Nq = t0->dims->data[1];
  const int C = t0->dims->data[2];  // num_classes+1 (includes background)
  if (B != 1) die("This demo expects B=1 outputs");
  if (args.doc_class_id < 0 || args.doc_class_id >= C) {
    die("--doc-class-id out of range: got " + std::to_string(args.doc_class_id) + ", C=" + std::to_string(C));
  }

  const int P2 = t_last->dims->data[2];
  if (P2 % 2 != 0) die("pred_points last dim must be even (x/y pairs)");
  const int P = P2 / 2;

  std::vector<float> logits_f32;
  std::vector<float> points_f32;

  auto read_out_f32 = [&](int out_tensor_index, std::vector<float>& dst) {
    const TfLiteTensor* t = interp->tensor(out_tensor_index);
    const int n = t->bytes / ((t->type == kTfLiteFloat16) ? 2 : 4);
    dst.resize(static_cast<size_t>(n));
    if (t->type == kTfLiteFloat32) {
      const float* p = interp->typed_tensor<float>(out_tensor_index);
      std::memcpy(dst.data(), p, dst.size() * sizeof(float));
    } else {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(t->data.raw_const);
      if (!p) die("Null output tensor buffer");
      for (int i = 0; i < n; ++i) dst[static_cast<size_t>(i)] = half_bits_to_float(p[i]);
    }
  };

  read_out_f32(outs[0], logits_f32);
  read_out_f32(outs.back(), points_f32);

  int best_q = 0;
  float best_score = -1.0f;
  for (int q = 0; q < Nq; ++q) {
    const float* row = logits_f32.data() + q * C;
    std::vector<float> prob = softmax(row, C);
    float s = prob[args.doc_class_id];
    if (s > best_score) {
      best_score = s;
      best_q = q;
    }
  }

  std::cout << "pred_logits shape: " << dims_to_string(t0) << "\n";
  std::cout << "pred_points shape: " << dims_to_string(t_last) << "\n";
  std::cout << "best query q*: " << best_q << " (softmax prob for doc_class_id=" << args.doc_class_id
            << " is " << best_score << ")\n";

  const float* qpts = points_f32.data() + best_q * (P * 2);
  std::cout << "points (normalized 0..1):\n";
  for (int i = 0; i < P; ++i) {
    float x = qpts[i * 2 + 0];
    float y = qpts[i * 2 + 1];
    std::cout << "  p" << i << ": (" << x << ", " << y << ")\n";
  }

  std::cout << "points (pixels in resized image " << W << "x" << H << "):\n";
  for (int i = 0; i < P; ++i) {
    float x = qpts[i * 2 + 0] * static_cast<float>(W);
    float y = qpts[i * 2 + 1] * static_cast<float>(H);
    std::cout << "  p" << i << ": (" << x << ", " << y << ")\n";
  }

  // Visualize: draw polygon from first 4 points (trained points; others may be -1 padded).
  if (!args.vis_out.empty()) {
    cv::Mat vis_rgb = load_rgb_resized_u8_with_opencv(args.image_path, H);
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (int i = 0; i < std::min(4, P); ++i) {
      const float xn = qpts[i * 2 + 0];
      const float yn = qpts[i * 2 + 1];
      if (xn < 0.0f || yn < 0.0f) continue;  // padding
      int px = static_cast<int>(std::lround(xn * static_cast<float>(W)));
      int py = static_cast<int>(std::lround(yn * static_cast<float>(H)));
      px = std::max(0, std::min(W - 1, px));
      py = std::max(0, std::min(H - 1, py));
      poly.emplace_back(px, py);
      cv::circle(vis_rgb, cv::Point(px, py), 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    if (poly.size() >= 3) {
      cv::polylines(vis_rgb, poly, true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
    cv::Mat vis_bgr;
    cv::cvtColor(vis_rgb, vis_bgr, cv::COLOR_RGB2BGR);
    {
      std::error_code ec;
      const std::filesystem::path out_p(args.vis_out);
      const std::filesystem::path parent = out_p.parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    }
    if (!cv::imwrite(args.vis_out, vis_bgr)) {
      die("Failed to write vis image: " + args.vis_out + " (check parent dir exists and is writable)");
    }
    std::cout << "saved visualization -> " << args.vis_out << "\n";
  }

  return 0;
}

