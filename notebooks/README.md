# Notebook Index

The project is currently notebook-first. Run the notebooks in the order below when reproducing the full workflow.

## 1. COCO JSON to Masks

Notebook: `coco_json_to_masks.ipynb`

Purpose:

- Loads COCO polygon annotations.
- Creates binary flood/water masks.
- Saves one mask per source image.
- Produces black masks for images without annotations.

Main values to update:

```python
coco = COCO("path/to/_annotations.coco.json")
img_dir = "path/to/images"
mask_dir = "path/to/output/masks"
```

## 2. U-Net Segmentation

Notebook: `unet_segmentation.ipynb`

Purpose:

- Defines the U-Net architecture.
- Loads image and mask pairs.
- Trains a binary flood/water segmentation model.
- Saves the model checkpoint.
- Runs visual single-image inference.

Main values to update:

```python
DATA_PATH = "path/to/segmentation/data"
MODEL_SAVE_PATH = "path/to/models/unet.pth"
```

## 3. YOLOv8 Detection

Notebook: `yolov8_detection.ipynb`

Purpose:

- Loads a YOLOv8 checkpoint.
- Trains or validates person detection on a YOLO-format dataset.
- Runs sample image prediction.

Main values to update:

```python
model = YOLO("path/to/yolo_checkpoint.pt")
results = model.train(data="path/to/data.yaml", epochs=100)
```

## 4. Final Life Detection Pipeline

Notebook: `final_life_detection_pipeline.ipynb`

Purpose:

- Runs U-Net segmentation on an input flood image.
- Runs YOLOv8 person detection on the same image.
- Scores each person box by flood-mask coverage.
- Saves an image with highlighted candidate detections.

Main values to update:

```python
SINGLE_IMG_PATH = "path/to/input_image.jpg"
MODEL_PATH = "path/to/unet.pth"
model = YOLO("path/to/yolo_best.pt")
```

## Run Notes

- Keep model checkpoints outside Git unless they are small and intentionally versioned.
- Keep datasets outside Git unless licensing allows redistribution.
- Record dataset version, training date, epoch count, and validation metrics for each experiment.
- Save qualitative output samples when dataset permissions allow public display.
