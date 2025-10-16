# MarI/O Kart - Mario Kart DS ML + Overlays

> **Train AI, render overlays, and read real game state from DeSmuME.**  
> A batteries-included toolkit for **Mario Kart DS** research and tooling: memory I/O, geometry/projection, collision & map parsers, and an extensible overlay system — with optional Torch-powered acceleration.

---

## Documentation
Docs and research articles can be found:
[https://gg-blake.github.io/marIOkart/](https://gg-blake.github.io/marIOkart)

---

## Quick Demo

##### Figure: Realtime overlay rendering using PyCairo + GTK (AI demo training in progress)
https://github.com/user-attachments/assets/ba1a553b-9bb7-4717-985b-65efe388cb1a

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Layout](#project-layout)
- [Quickstart](#quickstart)
  - [Run the emulator with overlays](#run-the-emulator-with-overlays)
  - [Add your own overlay](#add-your-own-overlay)
  - [Use the standalone `mkds` parser](#use-the-standalone-mkds-parser)
- [Architecture](#architecture)
- [Docs](#docs)
- [Controls](#controls)
- [Performance Notes](#performance-notes)
- [FAQ](#faq)
- [Legal](#legal)
- [License](#license)

---

## Overview

This repo brings together **three pillars**:

1. **Memory + Geometry utilities** — Read live MKDS game state from the **DeSmuME** emulator (position, camera, objects, checkpoints) and project world points to **screen space (256×192)** using a reconstructed model–view + perspective pipeline.
2. **Overlay system** — A queue-based, thread-safe drawing stack (Cairo/GTK) with a library of overlays for collision, checkpoints, objects, and camera targets — designed to be **extensible**.
3. **Parsers** — Clean readers for **KCL** (collision) and **NKM** (course map) formats; also available as an installable standalone lib `mkds` on PyPI.

> The long-term goal: **train agents** to drive in MKDS, using the overlay + geometry for interpretability and debugging.

---

## Features

- **DeSmuME integration (py-desmume)**: Read objects, player state, camera FOV/aspect/pose, clock, checkpoints.
- **Projection utilities**: Rebuild model–view and perspective matrices; project `(N,3)` world points → `(N,4)` screen points `[x, y, clip_z, depth_norm]`.
- **KCL reader**: Positions, normals, prisms, octree search; reconstruct triangle vertices.
- **NKM reader**: Objects, paths, checkpoints, cameras, respawns, etc.
- **Overlay framework**: Composable draw ops (`draw_points`, `draw_lines`, `draw_triangles`, `draw_paragraph`) enqueued from worker threads; Cairo renders on GTK draw.
- **Torch extensions (optional)**: Tensorized KCL/NKM variants for fast geometry and filtering (`is_wall`, `is_floor`, offroad masks, batched raycasts).
- **Training hooks**: Example scaffolding (work-in-progress) for agent training and fitness evaluation.

---

## Installation
Python ≥ 3.11 required, virtual environment is recommended 
### 1. Clone the repo
```bash
git clone https://github.com/gg-blake/marIOkart
cd marIOkart
```

### 2. Install System Dependencies
**Mac + Homebrew (recommended)**
```bash
brew install pygobject3 gtk4
```
**Linux (Ubuntu/Debian)**
```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0
```
> For other Linux distros consult the site [here](https://pygobject.gnome.org/getting_started.html).

**Windows**
> Windows is currently untested for this project, but you can try with installation guide [here](https://pygobject.gnome.org/getting_started.html).

### 3. Install Other Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. If using HOMEBREW, make sure that you have these environment variables set
```bash
export PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig:
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:
```

Platform-specific install commands vary; use your package manager (Homebrew, apt, pacman, etc.).

> The **parsers** are separately available as [`mkds`](https://pypi.org/project/mkds/):
>
> ```bash
> pip install mkds
> ```

This guide is a work in-progress, if you are having trouble with setup it to work on your machine, [message me on Discord](https://discord.com/users/410111287872323594), we'll get it fixed.

---

## Project Layout

```
.
├── mkds/                  # Standalone KCL/NKM parsing library (PyPI: mkds)
│   └── mkds/                  # Standalone KCL/NKM parsing library (PyPI: mkds)
│       ├── kcl.py
│       ├── nkm.py
│       └── utils.py
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory.py      # Emulator memory reads + camera & projection
│   │   ├── model.py       # Geometry helpers (rays, intersections, sampling)
│   │   └── train.py
│   ├── main.py            # GTK runner wiring emulator + overlays     
│   ├── misc/
│   │   └── courses.json
│   ├── mkds_extensions/
│   │   ├── kcl_torch.py   # Torch KCL (triangles, nearest queries)
│   │   └── nkm_torch.py   # Torch NKM (tensorized checkpoints)
│   ├── utils/
│   │   └── vector.py
│   └── visualization/
│       ├── draw.py        # Draw queue & Cairo primitives
│       └── overlay.py     # Built-in overlays (collision, checkpoints, etc.)
├── tests/                 # Various test scripts for experimentation
├── courses/               # Extracted course assets (KCL/NKM per course)               
└── README.md
```

---

## Quickstart

### Run the emulator with overlays

```bash
python main.py
```

- The GTK window shows the live emulator frame with overlays composited on top.
- Tweak the overlay list in `main.py`:

```python
from utils.overlay import (
    collision_overlay,
    checkpoint_overlay_1, checkpoint_overlay_2,
    player_overlay, raycasting_overlay, camera_overlay,
    stats_overlay,
)

run_emulator(
    generate_trainer,  # or a simple generator
    [
        collision_overlay,
        checkpoint_overlay_1,
        checkpoint_overlay_2,
        # player_overlay,
        # camera_overlay,
        # raycasting_overlay,
        # stats_overlay,
    ],
)
```

### Add your own overlay

An overlay is just a function: read → project → enqueue draws.

```python
# utils/my_overlay.py
import numpy as np
import torch
from utils.draw import draw_points
from utils.memory import read_position, project_to_screen, z_clip_mask

def my_overlay(emu, device=None):
    pos = read_position(emu, device=device).unsqueeze(0)      # (1,3)
    sp  = project_to_screen(emu, pos, device=device)          # (1,4) [x,y,clip_z,depth_norm]
    mask = z_clip_mask(sp)
    sp = sp[mask]
    if sp.shape[0] == 0:
        return
    pts = torch.cat([sp[:, :2], sp[:, 3, None]], dim=-1).cpu().numpy()
    draw_points(pts, colors=np.array([0.2, 0.8, 0.6]), radius_scale=6.0)
```

Then include `my_overlay` in the overlay list.

### Use the standalone `mkds` parser

```bash
pip install mkds
```

```python
from mkds.nkm import NKM
from mkds.kcl import KCL

nkm = NKM.from_file("courses/figure8/course_map.nkm")
print("Lap count:", nkm._STAG.amt_of_laps)

kcl = KCL.from_file("courses/figure8/course_collision.kcl")
print("Prisms:", len(kcl.prisms))
```

---

## Architecture

```text
               ┌───────────────────────────┐
               │      DeSmuME (Core)       │
               │  emu.cycle(), RAM, I/O    │
               └─────────────┬─────────────┘
                             │
memory.py (read_*  project_to|_screen, z_clip_mask, model-view)
                             │
               ┌─────────────┴─────────────┐
               │                           │
               │overlay.py(compute,enqueue)│
               │                           │
               │  mkds/(KCL/NKM parsers)   │
               │                           │
               └─────────────┬─────────────┘
                             │
       draw.py (thread-safe queue of Cairo closures)
                             │
        GTK draw event (main.py → consume_draw_stack)
                             │
            Composited overlay surface → window
```

- **Threading:** Overlays run in a worker thread and enqueue closures; the GTK thread executes them with a `cairo.Context`.
- **Projection:** Camera FOV/aspect + pose are read from memory to rebuild the perspective pipeline; screen origin is **top-left** (256×192).
- **KCL/NKM:** Binary readers expose structured data; Torch variants add vectorized geometry & filtering.

---

## Docs

Full, in-repo documentation (recommended to place under `docs/`):

- **Emulator I/O & Geometry Utilities** — API reference & examples  
  `READING_FILES.md`
- **Standalone Parsers (`mkds`)** — NKM & KCL readers (installable)  
  `READING_MEMORY.md`
- **Overlay System** — Draw queue design & built-in overlays  
  `OVERLAYS.md`

> These were generated from our discussion. If you want, regenerate them with your preferred doc tool (Sphinx, MkDocs, pdoc) or keep them as-is.

---

## Controls

Default keyboard mapping (via `pynput` → `py-desmume`):

| Key         | DS Button                |
|---          |---                       |
| W/A/S/D     | D-Pad Up/Left/Down/Right |
| Z / X       | B / A                    |
| U / I       | X / Y                    |
| Q / E       | L / R                    |
| Space       | Start                    |
| Arrow keys  | D-Pad                    |

---

## Performance Notes

- **Batch everything**: project arrays of points at once; avoid per-point projection.
- **Cull early**: use `z_clip_mask` before NumPy conversion.
- **Minimize host↔device hops**: keep tensors on device for math; convert to CPU right before draw.
- **Overlay surface cache**: the GTK draw path reuses the last overlay frame if no new draw ops were enqueued.
- **No anti-aliasing**: nearest-neighbor and `ANTIALIAS_NONE` preserve DS aesthetics and cost less.

---

## FAQ

**Q: Where do the memory addresses live?**  
A: In `utils/memory.py` (e.g., racer ptr `0x0217ACF8`, camera `0x0217AA4C`, etc.). The utilities hide fixed-point conversions and provide typed readers.

**Q: How do I filter walls vs floors?**  
A: Use KCL prism attributes. Torch variant exposes `kcl.prisms.is_wall` / `is_floor` masks.

**Q: Why are screen Y coordinates flipped?**  
A: DS origin is top-left; projection maps NDC accordingly with `(1 - y_ndc)` for Y.

**Q: Can I use this without Torch?**  
A: Yes. The I/O + parsers work without Torch; Torch is used for vectorized geometry and speed-ups.

**Q: Training status?**  
A: Scaffolding exists (`utils/train.py` + hooks). Model training demos are **in progress**.

---

## Legal

- You must **own the MKDS ROM** you use. This repo does **not** include game data or ROMs.
- DeSmuME and py-desmume are external projects; follow their licenses.

---

## License

See the repository’s `LICENSE` file for terms.
