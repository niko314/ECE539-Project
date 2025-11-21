# ECE539-Project

Full Pipeline Overview

1. Load COCO Annotations
   - Each person instance with valid keypoints becomes one training sample.

2. Lower-Body Keypoint Extraction
   - We select only 6 lower-body joints from the COCO 17-keypoint format.

3. Bounding Box Match
   - The person bbox is resized to match the model's aspect ratio (256×193).

4. Crop & Resize
   - The image is cropped using the matched bbox and resized to the model input size.

5. Keypoint Transformation
   - Ground-truth keypoints are transformed from original image coordinates → cropped coordinates → resized input coordinates.

6. Image Normalization
   - Images are normalized with ImageNet mean/std for compatibility with ResNet pretrained models.

7. Heatmap Target Generation
   - 6 Gaussian heatmaps(one per joint) are created at 1/4 resolution.

8. Model (PoseResNet18)
   - A ResNet-18 backbone with 3 deconv layers outputs heatmaps at 64×48 resolution.

9. Training
   - MSE loss over heatmaps
   - Adam optimizer
   - PCK@0.05 / PCK@0.10 computed every batch

10. Evaluation
    - Soft-argmax converts heatmaps → x,y predictions.
    - Validation PCK is monitored to save the best checkpoint.

