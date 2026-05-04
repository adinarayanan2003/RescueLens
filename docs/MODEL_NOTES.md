# Model Notes

## U-Net Segmentation

The segmentation notebook defines a standard encoder-decoder U-Net in PyTorch:

- Double convolution blocks
- Max-pooling downsampling
- Transposed-convolution upsampling
- Skip connections between encoder and decoder layers
- Single-channel output for binary segmentation

The model is trained with `BCEWithLogitsLoss`, which is appropriate for binary masks when the target mask is scaled to `[0, 1]`.

## YOLOv8 Detection

The detection notebook uses Ultralytics YOLOv8 for bounding-box detection. The training dataset should contain person labels in YOLO format.

The notebook currently loads a checkpoint from Google Drive and then runs validation, training, and prediction. Update the checkpoint and `data.yaml` paths before running.

## Combined Decision Logic

The final notebook combines both model outputs:

1. U-Net creates a binary flood/water mask.
2. YOLOv8 returns person bounding boxes.
3. The script measures mask coverage inside each bounding box.
4. Bounding boxes over the coverage threshold are highlighted.

## Important Assumptions

- YOLO boxes are aligned to the same image dimensions as the generated mask.
- U-Net mask output is resized to match the YOLO input/output image size.
- White or high-value mask pixels represent flood/water regions.
- A person box with enough flood-mask coverage is treated as a higher-risk detection.

## Known Technical Risks

- Resizing masks can introduce alignment errors if aspect ratios differ.
- Flood water can visually resemble road, mud, shadow, or reflections.
- People may be partially submerged, very small, occluded, or far from the camera.
- Training data quality strongly controls model behavior.

## Post-processing

The combined notebook uses area-overlap scoring:

```python
box_mask = flood_mask[y_min:y_max, x_min:x_max]
coverage = (box_mask > threshold).mean()

if coverage >= min_coverage:
    mark_as_high_risk()
```

This keeps the decision rule readable: each detected person receives a flood-overlap score, and only boxes over the chosen threshold are highlighted.
