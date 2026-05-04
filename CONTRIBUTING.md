# Contributing

Keep changes focused. This repo is intentionally notebook-first, so small, readable edits are better than a framework rewrite.

## What Belongs Here

- Notebook changes that improve the flood segmentation, person detection, or combined scoring flow.
- Documentation that helps someone reproduce a run.
- Metrics, sample outputs, and notes from real experiments.
- Small utilities that remove repeated notebook work without hiding the pipeline.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If training on GPU, install the correct PyTorch build for your CUDA version before running the notebooks.

## Pull Request Checklist

- Explain which notebook or pipeline stage changed.
- Document any new dataset layout requirements.
- Include before/after metrics when changing model logic.
- Include qualitative screenshots when changing inference or visualization.
- Do not commit large datasets, generated run folders, or model checkpoints unless intentionally required.
- Keep paths easy to change; avoid committing machine-specific absolute paths when a relative path will work.

## Experiment Reporting

When reporting model changes, include:

- Dataset version or source.
- Number of training, validation, and test images.
- Model checkpoint used for initialization.
- Epochs, batch size, image size, and learning rate.
- Detection metrics such as mAP, precision, and recall.
- Segmentation metrics such as IoU or Dice coefficient.
- Known failure cases.
