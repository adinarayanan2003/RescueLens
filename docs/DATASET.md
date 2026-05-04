# Dataset Notes

This project uses two related dataset formats: one for object detection and one for semantic segmentation.

## Detection Dataset

The YOLOv8 notebook expects a Roboflow/YOLO-style dataset with a `data.yaml` file.

Recommended structure:

```text
data/detection/flood_v8/
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
|-- test/
|   |-- images/
|   `-- labels/
`-- data.yaml
```

The `data.yaml` file should define the train, validation, and test image paths plus class names. For life/person detection, the important class is usually `person`.

## Segmentation Dataset

The U-Net notebook expects paired flood images and binary masks.

Recommended structure:

```text
data/segmentation/
|-- train/
|   |-- image_001.jpg
|   `-- image_002.jpg
|-- train_masks/
|   |-- mask_image_001.jpg
|   `-- mask_image_002.jpg
|-- test/
|   |-- image_101.jpg
|   `-- image_102.jpg
`-- test_masks/
    |-- mask_image_101.jpg
    `-- mask_image_102.jpg
```

The existing notebook currently uses folder names such as `test1`, `test1_masks`, `test`, and `test_masks`. Rename the folders or update the dataset class paths before training.

## COCO JSON to Mask Conversion

`notebooks/coco_json_to_masks.ipynb` converts COCO annotations into binary mask images.

Required inputs:

- COCO annotation file, usually named `_annotations.coco.json`
- Image directory containing all files referenced by the COCO JSON
- Output directory for generated masks

Output behavior:

- Annotated pixels are converted to white (`255`).
- Background pixels remain black (`0`).
- Images without annotations receive a black mask.

## Data Quality Checklist

- Image and mask filenames must sort in the same order or be matched explicitly.
- Each segmentation image should have exactly one corresponding mask.
- Masks should be single-channel binary images.
- Detection labels should use YOLO normalized coordinates.
- Train/validation/test splits should not contain duplicate images.
- Very small people, occluded people, and night/low-light scenes should be represented if they are expected during inference.
