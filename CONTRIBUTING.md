# Contributing

Contributions should make the project easier to reproduce, evaluate, or extend.

## Good First Improvements

- Replace hardcoded Colab paths with configurable path variables.
- Rename `CarvanaDataset` to a flood-specific dataset class.
- Add a proper train/validation/test metrics table to the README.
- Add example output images if dataset permissions allow it.
- Move reusable code from notebooks into a `src/` package.
- Replace bounding-box corner checks with mask area-overlap scoring.

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
- Keep paths portable; avoid committing user-specific absolute paths.

## Experiment Reporting

When reporting model changes, include:

- Dataset version or source.
- Number of training, validation, and test images.
- Model checkpoint used for initialization.
- Epochs, batch size, image size, and learning rate.
- Detection metrics such as mAP, precision, and recall.
- Segmentation metrics such as IoU or Dice coefficient.
- Known failure cases.
