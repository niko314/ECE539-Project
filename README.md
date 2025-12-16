# 🚀 How to run on Google Colab

This section explains how to run the full training pipeline on **Google Colab**, including environment setup, dataset download, subset generation, and model training.


You should change runtime with A100 gpu or L4 gpu or T4 gpu in Colab.

---

## **1. Install Dependencies**

Install PyTorch (CUDA 12.1), COCO API, and other required libraries.

```bash
%pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
%pip -q install pycocotools opencv-python tqdm matplotlib pillow
```

---

## **2. Clone the Repository**

Clone the project and add it to the Python path so Colab can import project modules.

```bash
!git clone https://github.com/daeyeon-kim-99/CS566_Project.git

import sys, torch, torch.backends.cudnn as cudnn
sys.path.append('/content/CS566_Project')
```

---

## **3. Download the COCO Dataset**

This project requires the **COCO 2017 train and val sets** for keypoint annotations.

```bash
%mkdir -p /content/CS566_Project/data/coco/annotations
%mkdir -p /content/CS566_Project/data/coco/images/val2017

%cd /content/CS566_Project/data/coco/annotations
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

%cd /content/CS566_Project/data/coco/images
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/zips/val2017.zip

!unzip train2017.zip
!unzip val2017.zip

%cd ..
%cd annotations/
!unzip annotations_trainval2017.zip

%cd /content/CS566_Project/
```

---

## **4. Generate Custom Datasets**

Our project uses two optional datasets besides the full COCO set:

1. **20K Mini Subset** – Faster training for testing
2. **Lower-Body Full-Visible Set (~60K)** – Only samples where 6 lower-body joints are fully visible

Generate them with:

```bash
!python coco_subset_20K.py
!python coco_lower_full_vis.py
```

---

## **5. Choose Dataset and Model in `train.py`**

You must manually update `train.py` depending on which dataset you want to use(line 27).

### **A) Choose Dataset**

```python
# Default full COCO dataset:
TRAIN_ANN = 'data/coco/annotations/annotations/person_keypoints_train2017.json'

# Use 20K subset:
TRAIN_ANN = 'data/coco_mini/person_keypoints_train2017_20k.json'

# Use lower-body full-visible dataset:
TRAIN_ANN = 'data/coco_lower_full/person_keypoints_train2017_lower_full.json'
```

---

### **B) Select Model Architecture**

Inside the model-building section(line 362), you can choose model:

```python
# Options:
# PoseResNet18
model = PoseResNet18(...)

# PoseResNet18FPN
model = PoseResNet18FPN(...)

# LowerBodyPoseNet (custom model)
model = LowerBodyPoseNet(...)
```

---

### **C) Adjust Hyperparameters**

```python
#At line 48-50
NUM_EPOCHS = 60     # You may increase to 80 or 120
LR = ...            # Learning rate
```

---

## **6. Run Training**

Once everything is configured, run:

```bash
!python train.py
```

**Let's Train!**

* Starts training using your selected model and dataset
* Saves checkpoints and logs to `out_lower/` or the specified output directory

---
