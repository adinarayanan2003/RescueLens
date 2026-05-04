# Computer Vision Guided Life Detection From Flood

Computer vision pipeline for identifying people in flood imagery and highlighting detections that overlap with flood/water regions. The project combines:

- **YOLOv8 object detection** to locate people in disaster/flood scene images.
- **U-Net semantic segmentation** to estimate flood/water regions.
- **Post-processing logic** to compare person bounding boxes against the predicted flood mask and flag high-risk detections.

The current implementation is notebook-first and designed for Google Colab experimentation.

## Problem Statement

Flood response teams often need to scan large image sets quickly to find people who may be stranded or exposed to flood water. This repository explores a two-stage computer vision approach:

1. Detect humans in flood-scene images.
2. Segment flood/water regions.
3. Mark detected humans whose bounding boxes intersect likely flood regions.

This can support triage workflows where responders review candidate images instead of manually scanning every frame.

## Repository Structure

```text
.
|-- notebooks/
|   |-- coco_json_to_masks.ipynb
|   |-- unet_segmentation.ipynb
|   |-- yolov8_detection.ipynb
|   `-- final_life_detection_pipeline.ipynb
|-- docs/
|   |-- DATASET.md
|   |-- WORKFLOW.md
|   `-- MODEL_NOTES.md
|-- CONTRIBUTING.md
|-- requirements.txt
`-- README.md
```

## Pipeline Overview

```text
Flood image
   |
   |-- U-Net segmentation --> Binary flood/water mask
   |
   |-- YOLOv8 detection ----> Person bounding boxes
                                |
                                v
                   Mask + box overlap check
                                |
                                v
                   Highlight people in flood regions
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/coco_json_to_masks.ipynb` | Converts COCO polygon annotations into binary mask images for segmentation training. |
| `notebooks/unet_segmentation.ipynb` | Defines a PyTorch U-Net model, dataset loader, training loop, validation loop, and single-image inference. |
| `notebooks/yolov8_detection.ipynb` | Uses Ultralytics YOLOv8 to train, validate, and run object detection on flood images. |
| `notebooks/final_life_detection_pipeline.ipynb` | Runs U-Net inference and YOLO inference on the same image, then checks whether detected person boxes touch likely flood regions. |

## Setup

### Option 1: Google Colab

The notebooks currently use Colab paths such as `/content/drive/MyDrive/project/...`.

1. Open a notebook from the `notebooks/` directory in Google Colab.
2. Mount Google Drive when prompted.
3. Upload or place datasets and model checkpoints in your Drive.
4. Update the path constants inside each notebook to match your Drive folder.
5. Run the notebook cells in order.

### Option 2: Local Python Environment

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation selector before running training.

## Expected Data Layout

The notebooks assume image, mask, and model paths are configured manually. A clean local layout is:

```text
data/
|-- detection/
|   `-- flood_v8/
|       |-- train/
|       |-- valid/
|       |-- test/
|       `-- data.yaml
|-- segmentation/
|   |-- train/
|   |-- train_masks/
|   |-- test/
|   `-- test_masks/
`-- raw_annotations/
    `-- _annotations.coco.json

models/
|-- yolo_best.pt
`-- unet.pth

outputs/
|-- masks/
|-- detections/
`-- highlighted/
```

See [docs/DATASET.md](docs/DATASET.md) for dataset details.

## How to Run

### 1. Generate Segmentation Masks

Use `notebooks/coco_json_to_masks.ipynb` to convert COCO annotations into binary masks.

Update:

```python
coco = COCO("path/to/_annotations.coco.json")
img_dir = "path/to/images"
mask_dir = "path/to/output/masks"
```

### 2. Train U-Net Segmentation

Use `notebooks/unet_segmentation.ipynb`.

Update:

```python
DATA_PATH = "path/to/segmentation/data"
MODEL_SAVE_PATH = "path/to/models/unet.pth"
```

The notebook trains a U-Net model using `BCEWithLogitsLoss` and saves a `.pth` checkpoint.

### 3. Train YOLOv8 Detection

Use `notebooks/yolov8_detection.ipynb`.

Update:

```python
results = model.train(data="path/to/data.yaml", epochs=100)
```

For inference, load the trained checkpoint:

```python
model = YOLO("path/to/yolo_best.pt")
model.predict("path/to/image.jpg", save=True)
```

### 4. Run the Combined Pipeline

Use `notebooks/final_life_detection_pipeline.ipynb`.

Update:

```python
SINGLE_IMG_PATH = "path/to/flood/image.jpg"
MODEL_PATH = "path/to/unet.pth"
model = YOLO("path/to/yolo_best.pt")
```

The final notebook:

1. Generates a flood/water mask for the input image.
2. Detects people using YOLOv8.
3. Checks bounding box corner pixels against the predicted mask.
4. Saves an output image with highlighted detections.

## Key Dependencies

- Python 3.10+
- PyTorch
- Torchvision
- Ultralytics YOLOv8
- OpenCV
- Pillow
- Matplotlib
- NumPy
- pycocotools

## Current Limitations

- Paths are still notebook-local and should be updated before each run.
- The U-Net dataset class is named `CarvanaDataset` even though it is used for flood segmentation.
- The final overlap logic checks bounding-box corner pixels only; a more robust approach would calculate the mask coverage ratio inside each bounding box.
- Model checkpoints and datasets are not included in the repository.
- Training metrics, confusion matrices, and qualitative output samples should be added after running the notebooks on the final dataset.

## Recommended Improvements

- Move reusable model, dataset, inference, and post-processing code into a `src/` package.
- Add command-line scripts for training and inference.
- Replace hardcoded Colab paths with a config file.
- Track experiment metrics in `runs/` or a lightweight experiment log.
- Add example output images once dataset permissions are confirmed.
- Improve the final risk scoring from corner checks to area-overlap scoring.

## Documentation

- [Dataset Notes](docs/DATASET.md)
- [Workflow](docs/WORKFLOW.md)
- [Model Notes](docs/MODEL_NOTES.md)
- [Notebook Index](notebooks/README.md)
- [Contributing](CONTRIBUTING.md)

## Project Status

This is an academic/prototype computer vision project. It demonstrates the core idea, but it should be validated carefully before any operational rescue or emergency-response use.
