# Computer Vision Guided Life Detection From Flood

Prototype pipeline for finding people in flood imagery and marking detections that sit inside predicted flood/water regions.

The workflow uses:

- YOLOv8 for person detection.
- A PyTorch U-Net for flood/water segmentation.
- Mask-overlap scoring to highlight detected people in flooded areas.

## Problem Statement

Flood imagery can contain many frames where only a few show people in risky positions. This project keeps the detection problem simple and inspectable:

1. Detect humans in flood-scene images.
2. Segment likely flood/water pixels.
3. Rank or mark detections by how much of the person box overlaps the flood mask.

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
                   Mask overlap scoring
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
| `notebooks/final_life_detection_pipeline.ipynb` | Runs U-Net and YOLO on the same image, then scores each person box by flood-mask coverage. |

## Setup

### Google Colab

The notebooks were developed in Colab. Set the path constants in each notebook to match your Drive or local folder layout.

1. Open a notebook from the `notebooks/` directory in Google Colab.
2. Mount Google Drive when prompted.
3. Upload or place datasets and model checkpoints in your Drive.
4. Update the path constants inside each notebook to match your Drive folder.
5. Run the notebook cells in order.

### Local Python Environment

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation selector before running training.

## Expected Data Layout

Keep datasets, model weights, and generated runs outside Git. The ignored local layout below works well with the notebooks:

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
3. Calculates flood-mask coverage inside each detected person box.
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

## Artifacts

Datasets and trained weights are intentionally not committed. Put them under `data/` and `models/` locally, or point the notebook constants to the corresponding Drive paths in Colab.

When publishing results, add the dataset version, checkpoint name, validation metrics, and a few qualitative outputs so the run can be traced later.

## Documentation

- [Dataset Notes](docs/DATASET.md)
- [Workflow](docs/WORKFLOW.md)
- [Model Notes](docs/MODEL_NOTES.md)
- [Notebook Index](notebooks/README.md)
- [Contributing](CONTRIBUTING.md)

## Project Status

Notebook-first research prototype. The code is kept small so the full pipeline can be inspected and modified quickly.
