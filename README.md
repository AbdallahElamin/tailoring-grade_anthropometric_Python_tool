# TailorVision

**Professional-grade anthropometric body measurement extraction from two client photos.**

Built on [SMPL-X](https://github.com/vchoutas/smplx) and [SMPL-Anthropometry](https://github.com/DavidBoja/SMPL-Anthropometry). Designed for traditional garment tailoring — outputs body measurements in centimetres with per-measurement confidence and tailoring-specific ease values.

---

## Features

- ✅ Front + side image input → 16 standard body measurements in cm
- ✅ PyTorch-based SMPL-X shape fitting (dual-view reprojection + anthropometric priors)
- ✅ Three scale modes: known height (best), heuristic, normalized fallback
- ✅ Monte-Carlo uncertainty estimates per measurement
- ✅ Garment-type ease tables: traditional, suit, shirt, trousers
- ✅ Structured JSON output with warnings and quality scores
- ✅ Click CLI + Python API
- ✅ Fully typed, modular, independently testable

---

## Measurements Extracted

| Label | Measurement |
|-------|-------------|
| A | head circumference |
| B | neck circumference |
| C | shoulder to crotch height |
| D | chest circumference |
| E | waist circumference |
| F | hip circumference |
| G | wrist right circumference |
| H | bicep right circumference |
| I | forearm right circumference |
| J | arm right length |
| K | inside leg height |
| L | thigh left circumference |
| M | calf left circumference |
| N | ankle left circumference |
| O | shoulder breadth |
| P | height |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/tailorvision
cd tailorvision
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
# or install as a package:
pip install -e .
```

> **GPU support:** replace the `torch` line in `requirements.txt` with:
> ```
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Then set `device = "cuda"` in your config.

### 3. Download SMPL-X model files *(required)*

Register (free) at https://smpl-x.is.tue.mpg.de and download the model files.  
Place them in:

```
models/
└── smplx/
    ├── SMPLX_MALE.npz
    ├── SMPLX_FEMALE.npz
    └── SMPLX_NEUTRAL.npz
```

### 4. Clone SMPL-Anthropometry *(required)*

```bash
git clone https://github.com/DavidBoja/SMPL-Anthropometry third_party/SMPL-Anthropometry
```

The measurement engine automatically adds this to `sys.path` at runtime.

---

## Usage

### Python API

```python
from tailorvision import TailorVisionPipeline, PipelineConfig

config = PipelineConfig(
    known_height_cm=175.0,   # optional but improves accuracy significantly
    gender="male",
    garment_type="traditional",
)

pipeline = TailorVisionPipeline(config)
result = pipeline.run("client_front.jpg", "client_side.jpg")

# Access measurements
print(result.measurements_cm)
# {'height': 175.0, 'chest_circumference': 96.4, 'waist_circumference': 82.1, ...}

# Access tailoring recommendations
print(result.tailoring_recommendations.chest_with_ease_cm)
# 108.4

# Save as JSON
result.save_json("output/client_001.json")
```

### CLI

```bash
tailor-vision measure \
  --front  client_front.jpg \
  --side   client_side.jpg \
  --height 175 \
  --gender male \
  --garment traditional \
  --output output/client_001.json \
  --verbose
```

### Via `python -m`

```bash
python -m tailorvision measure --front front.jpg --side side.jpg --height 175
```

---

## Output JSON Schema

```json
{
  "body_model_type": "smplx",
  "gender": "male",
  "smplx_parameters": { "betas": [...], "pose_neutralized": true },
  "measurements_cm": {
    "height": 175.2,
    "chest_circumference": 96.4,
    "waist_circumference": 82.0,
    "hip_circumference": 99.2,
    "shoulder_breadth": 44.5,
    "arm_right_length": 62.1,
    "inside_leg_height": 79.3
  },
  "measurement_confidence": {
    "chest_circumference": "HIGH",
    "waist_circumference": "MEDIUM"
  },
  "uncertainty_cm": { "waist_circumference": 2.1 },
  "scale": { "mode": "known_height", "scale_factor": 1.043, "confidence": 0.97 },
  "quality_scores": { "overall": 0.79 },
  "warnings": [],
  "tailoring_recommendations": {
    "garment_type": "traditional",
    "chest_with_ease_cm": 108.4,
    "waist_with_ease_cm": 96.0,
    "collar_size_cm": 40.5,
    "rise_cm": 27.8,
    "sleeve_length_cm": 64.1
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests that do **not** require model files (pure-logic tests):
- `test_validator.py` — image quality gate
- `test_scale_recovery.py` — scale modes with mock adapter
- `test_tailoring_mapper.py` — ease allowances and derived values
- `test_pose_estimator.py` — stub pose estimator
- `test_keypoint_lifter.py` — biview fusion
- `test_schema.py` — Pydantic JSON round-trip

---

## Architecture

```
tailorvision/
├── __init__.py          ← Public API: TailorVisionPipeline, PipelineConfig
├── pipeline.py          ← 8-stage orchestrator
├── config.py            ← PipelineConfig dataclass (all knobs here)
├── schema.py            ← Pydantic v2 output models
├── exceptions.py        ← Typed exception hierarchy
│
├── input/               ← Stage 1–2: validation & preprocessing
│   ├── loader.py        ← load_image(), EXIF correction, metadata
│   ├── validator.py     ← QualityGate: resolution, blur, completeness
│   └── preprocessor.py ← Resize, optional BG removal
│
├── vision/              ← Stage 3: pose & segmentation
│   ├── pose_estimator.py ← PoseEstimator protocol + MediaPipe backend
│   ├── segmentor.py     ← Body silhouette extraction
│   └── keypoint_lifter.py ← Fuse front+side → BiViewPose
│
├── fit/                 ← Stage 4–5: SMPL-X shape fitting
│   ├── body_model_adapter.py ← Wraps smplx.create(), T-pose vertices
│   ├── pose_fit_engine.py    ← PyTorch Adam optimiser over betas
│   └── anthropometric_prior.py ← Differentiable proportion priors
│
├── scale/               ← Stage 6: metric scale recovery
│   └── scale_recovery_engine.py ← Known height / heuristic / normalised
│
├── measure/             ← Stage 7: measurement extraction
│   ├── measurement_engine.py ← Wraps SMPL-Anthropometry MeasureBody
│   └── uncertainty.py        ← Monte-Carlo ±σ estimation
│
├── tailor/              ← Stage 8a: garment mapping
│   ├── ease_tables.py   ← Per-garment ease allowance tables
│   └── tailoring_mapper.py ← Applies ease, computes derived values
│
├── quality/             ← Stage 8b: QA reporting
│   └── quality_reporter.py ← Aggregates scores, emits warnings
│
└── api/                 ← CLI and programmatic interface
    └── cli.py           ← Click CLI (tailor-vision measure ...)

third_party/
└── SMPL-Anthropometry/  ← Clone here (git clone ...)

models/
└── smplx/               ← Place SMPLX_*.npz files here

tests/
├── test_validator.py
├── test_scale_recovery.py
├── test_tailoring_mapper.py
├── test_pose_estimator.py
├── test_keypoint_lifter.py
└── test_schema.py
```

---

## Accuracy Notes

| Scenario | Expected Accuracy |
|---|---|
| Known height + clear photos + fitted clothes | ±1–2 cm on circumferences |
| Known height + moderate clothing | ±2–4 cm |
| Unknown height, heuristic scale | ±5–15% relative (all measurements scale together) |
| Loose/baggy clothing | Circumferences may be inflated 2–8 cm; warning emitted |
| Non-upright pose | Lengths may be compressed; BAD_POSTURE warning |

**Sub-centimetre accuracy is not claimed** unless `known_height_cm` is provided,
the person wears fitted clothing, and the pose quality score is HIGH.

---

## Photo Guidelines (for best results)

1. Stand against a plain, light-coloured wall.
2. Wear fitted clothing (leggings, fitted shirt).
3. Arms slightly away from the body, feet shoulder-width apart.
4. Capture the entire body from head to feet.
5. Natural lighting, no strong shadows.
6. Camera at chest height, ~2–3 metres away.
7. Front view: face the camera directly.
8. Side view: turn exactly 90° to the right.

---

## Licensing

- **SMPL-X model files**: Non-commercial research license — see https://smpl-x.is.tue.mpg.de
- **SMPL-Anthropometry**: MIT License
- **This codebase**: MIT License

> ⚠️ For commercial deployment, obtain a commercial license from the Max Planck Institute
> for Intelligent Systems before using SMPL-X model files.

---

## Citation

If you use this system in research, please cite:

```bibtex
@inproceedings{SMPL-X:2019,
  title  = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and others},
  booktitle = {CVPR},
  year   = {2019}
}
```
