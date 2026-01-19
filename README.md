# Benchmark Pipeline (Torch / ONNX Runtime / OpenVINO / TensorRT)

A unified benchmarking pipeline for deep learning inference across heterogeneous hardware backends (CPU / GPU / NPU / DLA).

This project is designed for **engineering-grade measurements**, reproducible artifacts, and conservative validation.  
It intentionally avoids dynamic model construction and focuses on benchmarking **pre-exported, initialized model artifacts**.

---

## Key Features

### Backends
- PyTorch
- ONNX Runtime (CPU / CUDA / TensorRT EP where available)
- OpenVINO (CPU / GPU / NPU, including AUTO / HETERO policies)
- TensorRT (primarily Linux / Jetson workflows)

### Metrics
- Latency distribution (mean, p50, p90, p95, p99, raw samples)
- Throughput (samples/s)
- CPU utilization sampling (with sampling diagnostics)
- Memory tracking (process RSS over time and peak)
- Optional model-level metrics (MACs, model size) where supported

### Outputs
- Stable, timestamped run directories under `runs/`
- JSON as the primary artifact (optional CSV)
- Optional plots and Markdown reports via CLI flags

### Quality & Safety
- Unit tests, integration tests (backend-gated)
- Golden regression tests validate **schema structure**, not numeric values
- Device and path sanitization to prevent unsafe filesystem usage
- Internal or proprietary model source code can be kept **out of Git**

---

## Repository Layout

```text
benchmark-pipeline/
├─ bench/
│  ├─ cli/                       # CLI entry points (single-run, multi-run)
│  ├─ core/                      # Core pipeline (runners, metrics, schemas)
│  ├─ configs/
│  │  ├─ examples/               # Documented config templates
│  │  └─ local/                  # Machine-specific configs (not committed)
│  ├─ tests/                     # Unit, integration, golden tests
│  └─ tools/                     # Utility scripts (e.g. model export)
├─ docs/                         # Detailed documentation
├─ runs/                         # Benchmark outputs (not committed)
├─ pyproject.toml
└─ README.md

