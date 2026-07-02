# WorldMirror on legacy CUDA GPUs (Maxwell / sm_52)

This fork makes HunyuanWorld-Mirror inference run on pre-Volta NVIDIA GPUs
(e.g. Tesla M40, GTX 900/1000 series) that upstream can't use because of two
hard assumptions:

| Upstream assumption | Why it breaks on Maxwell | Fix in this fork |
|---|---|---|
| `bfloat16` autocast hardcoded in `visual_transformer.py` | bf16 needs Ampere (sm_80); fp16 GEMMs need sm_53 — Maxwell M40 is sm_52 | `_amp_autocast()` helper: bf16 where supported, full-precision fallback otherwise |
| fp16 autocast in `worldmirror.py::prepare_contexts` | same | gated on `torch.cuda.is_bf16_supported()` |
| `gsplat` imported at module level | gsplat kernels require compute capability >= 7.0 | lazy/optional import; splat prediction + PLY export work without it, only novel-view `rendered.mp4` is skipped |

## Install (tested: Tesla M40 12GB, driver 535, CUDA 12.x)

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121  # last line with sm_52 kernels
pip install -r requirements_m40.txt
```

## Run (video -> 3D world)

```bash
python infer.py --input_path your_clip.mp4 --fps 1 --output_path out
```

Outputs in `out/`: `gaussians.ply` (3DGS world — view it with any splat viewer,
e.g. Spark/Three.js, SuperSplat), `points.ply`, per-frame depth + normal maps,
camera poses (COLMAP sparse).

Notes for 11.5GB VRAM:
- The checkpoint is 5.05GB; loaded fp32 it leaves ~6GB for activations.
- Keep the frame count modest (`--fps 1`, clips <= ~16 frames) and
  `--target_size 518`. More frames = quadratic global-attention memory.
- Everything runs fp32 (no tensor cores on Maxwell), so expect minutes per
  scene, not the 4s an A100 gets. It completes — that's the point.
