# Workflow

This repository is organized around an end-to-end computer vision experiment for flood-scene life detection.

## 1. Prepare Data

Collect flood-scene images and annotations:

- Bounding boxes for people, used by YOLOv8.
- Segmentation polygons or masks for flood/water regions, used by U-Net.

If flood/water annotations are in COCO format, run `notebooks/coco_json_to_masks.ipynb` to generate binary masks.

## 2. Train Flood Segmentation

Run `notebooks/unet_segmentation.ipynb`.

The notebook:

1. Defines a PyTorch U-Net architecture.
2. Loads paired image and mask data.
3. Splits data into training and validation sets.
4. Trains using `BCEWithLogitsLoss`.
5. Saves the trained `.pth` model checkpoint.
6. Runs single-image mask inference for visual inspection.

Expected output: `models/unet.pth`

## 3. Train Person Detection

Run `notebooks/yolov8_detection.ipynb`.

The notebook:

1. Installs Ultralytics.
2. Loads a YOLOv8 model/checkpoint.
3. Validates the model.
4. Trains on a YOLO-format dataset.
5. Runs prediction on a sample flood image.

Expected output: `models/yolo_best.pt` or a YOLO run directory containing `best.pt`.

## 4. Run Combined Inference

Run `notebooks/final_life_detection_pipeline.ipynb`.

The notebook:

1. Loads the trained U-Net checkpoint.
2. Predicts a binary flood/water mask for the input image.
3. Loads the trained YOLOv8 checkpoint.
4. Detects people in the same input image.
5. Checks whether person detections overlap likely flood regions.
6. Saves an image with highlighted bounding boxes.

Expected output: highlighted candidate detections in `outputs/` or the configured notebook path.

## 5. Review Results

Review outputs manually and record:

- True positives: people correctly detected in flood/water regions.
- False positives: non-risk detections incorrectly highlighted.
- False negatives: people missed by YOLO or missed by the overlap check.
- Segmentation failures: water regions missing or background marked as water.

## Suggested Evaluation Metrics

Detection:

- mAP@50
- mAP@50-95
- Precision
- Recall

Segmentation:

- IoU
- Dice coefficient
- Pixel accuracy

Combined pipeline:

- Person-in-flood precision
- Person-in-flood recall
- False alarm rate per image
- Manual review time saved
