# Body3D Reconstruction

A modular Python framework for 3D human body reconstruction from RGB-D data,
automated anthropometric measurement extraction, and clinical volumetric
analysis using the SMPL body model.

---

## Overview

This system captures synchronized RGB and depth frames from an Intel RealSense
D455 camera, reconstructs a 3D body mesh, extracts anthropometric measurements,
and compares the result against a synthetic reference generated from a
multivariate regression model. The comparison is visualized as a color-coded
3D mesh and exported as a clinical PDF report.

---

## Features

- Real-time RGB + Depth stream from Intel RealSense D455
- Simulation mode for development without hardware
- Body segmentation and 3D point cloud reconstruction
- SMPL body model fitting with automated beta optimization
- Multivariate regression model for synthetic reference generation
- Volumetric comparison: real mesh vs synthetic reference (red = excess, blue = deficit)
- 4-view visualization: frontal, lateral right, posterior, lateral left
- Anthropometric measurements: neck, chest, waist, hip, thigh, knee, wrist
- Clinical PDF report with patient data, measurements, and interpretation
- SMPL optimization cache (first run ~40s, subsequent runs <2s)
- Import RGB + depth (.npy) from disk
- PyQt5 GUI with zoom, pan, and re-execution support

---

## Project Structure

```
body3d-reconstruction/
├── app.py                      # GUI entry point
├── main.py                     # Pipeline entry point (terminal)
├── requirements.txt
├── README.md
├── LICENSE
│
├── src/
│   ├── camera.py               # RealSense D455 + simulation mode
│   ├── loader.py               # RGB + depth loading
│   ├── preprocessing.py        # Depth filtering, hole filling, smoothing
│   ├── segmentation.py         # Body segmentation from depth
│   ├── reconstruction.py       # Point cloud reconstruction
│   ├── measurements.py         # Anthropometric measurement extraction
│   ├── smpl_fitting.py         # SMPL model fitting + beta optimization
│   ├── smpl_cache.py           # Beta cache to avoid reoptimization
│   ├── regression_model.py     # Multivariate regression model
│   ├── volume_comparison.py    # Volumetric comparison + figure generation
│   ├── morphing.py             # SMPL mesh morphing utilities
│   ├── visualization.py        # Matplotlib visualization utilities
│   ├── pdf_report.py           # Clinical PDF report generation
│   ├── data_inspector.py       # Data inspection utilities
│   └── gui/
│       ├── __init__.py
│       └── main_window.py      # PyQt5 main window
│
├── models/
│   └── smpl/
│       ├── SMPL_NEUTRAL.pkl    # SMPL neutral model (not tracked by git)
│       ├── SMPL_MALE.pkl
│       └── SMPL_FEMALE.pkl
│
├── data/
│   ├── sample/
│   │   ├── rgb.png             # Sample RGB frame
│   │   └── depth.npy           # Sample depth map (uint16, mm units)
│   └── captured/               # Frames captured from camera
│
└── output/                     # Generated figures, reports, cache
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gaby-calypso/body3d-reconstruction.git
cd body3d-reconstruction
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt --break-system-packages
```

### 3. Install RealSense SDK (optional — only needed with physical camera)

```bash
pip3 install pyrealsense2 --break-system-packages
```

### 4. Download SMPL models

Download the SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)
and place them in `models/smpl/`:

```
models/smpl/SMPL_NEUTRAL.pkl
models/smpl/SMPL_MALE.pkl
models/smpl/SMPL_FEMALE.pkl
```

> SMPL model files are not included in this repository due to license restrictions.

---

## Usage

### GUI application

```bash
python3 app.py
```

1. Press **▶ Iniciar** to start the camera stream (auto-detects RealSense or falls back to simulation)
2. Press **⬤ Capturar frame** to capture a frame, or **📂 Importar imágenes** to load from disk
3. Fill in patient parameters: body fat %, sex, age, weight, height
4. Press **⚡ Ejecutar pipeline** to run the full pipeline
5. Press **⬇ Exportar PDF clínico** to save the clinical report

### Terminal pipeline

```bash
python3 main.py
```

---

## Input Data Format

| File      | Format          | Units                                |
| --------- | --------------- | ------------------------------------ |
| RGB image | `.png` / `.jpg` | —                                    |
| Depth map | `.npy` (uint16) | millimeters (RealSense D455 default) |

---

## Regression Model

Anthropometric measurements are predicted from user-provided parameters using
a multivariate linear regression model calibrated on an external dataset:

| Measurement | Predictors                           |
| ----------- | ------------------------------------ |
| Neck        | Body fat %, sex, age, weight, height |
| Chest       | Body fat %, sex, age, weight, height |
| Abdomen     | Body fat %, sex, age, weight, height |
| Hip         | Body fat %, sex, age, weight, height |
| Knee        | Body fat %, sex, age, weight, height |
| Thigh       | Body fat %, sex, age, weight, height |
| Wrist       | Body fat %, sex, age, weight, height |

---

## Volumetric Comparison

The system overlays the real SMPL mesh (color-coded) on the synthetic
reference mesh (gray) and computes per-vertex signed distances:

- **Red** — real body volume exceeds reference (excess)
- **Blue** — real body volume is below reference (deficit)
- **White** — no significant difference

Results are shown in 4 views: frontal, right lateral, posterior, left lateral.

---

## Requirements

- Python 3.11+
- macOS / Linux (tested on macOS)
- Intel RealSense D455 (optional — simulation mode available)
- 8 GB RAM minimum (16 GB recommended for SMPL optimization)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [SMPL](https://smpl.is.tue.mpg.de) — Skinned Multi-Person Linear Model
- [smplx](https://github.com/vchoutas/smplx) — SMPL-X Python library
- [Open3D](http://www.open3d.org) — 3D data processing
- [Intel RealSense](https://www.intelrealsense.com) — RGB-D camera
