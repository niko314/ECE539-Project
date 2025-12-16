import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# For mini dataset(20K), you should change path
# TRAIN_ANN to 'data/coco_mini/person_keypoints_train2017_20k.json'

# For lower full visible dataset(60K), you should change path
# TRAIN_ANN to 'data/coco_lower_full/person_keypoints_train2017_lower_full.json'

TRAIN_ANN = 'data/coco/annotations/annotations/person_keypoints_train2017.json'
VAL_ANN   = 'data/coco/annotations/annotations/person_keypoints_val2017.json'
TRAIN_IMG_DIR = 'data/coco/images/train2017'
VAL_IMG_DIR = 'data/coco/images/val2017'

# input height, width
IN_H, IN_W = 256, 192

# heatmap height, width
# since the ouput of model after deconv, the height, weight is H/4, W/4
HM_H, HM_W = IN_H // 4, IN_W // 4
RATIO = IN_W / IN_H
SIGMA = 2.0

# lower-body joint index (COCO 17 keypoints)
LOWER_IDX = [11, 13, 15, 12, 14, 16]  # [Lhip, Lknee, Lankle, Rhip, Rknee, Rankle]
FLIP_PAIRS = [(0, 3), (1, 4), (2, 5)] # Flip (L<->R)
NUM_JOINTS = 6

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EPOCHS = 10

LR = 0.001
WD = 0.0001

BBOX_SCALE = 1.0

# checkpoint path
RUN_TAG = f"e{EPOCHS}"
OUT_DIR = Path(f'out_lower/{RUN_TAG}')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_target_heatmaps(joints, joints_vis, num_joints=NUM_JOINTS,
                             hm_size=(HM_W, HM_H), sigma=SIGMA):
    W, H = hm_size
    target = np.zeros((num_joints, H, W), dtype=np.float32)
    target_weight = np.zeros((num_joints, 1), dtype=np.float32)

    kernel_radius = sigma * 3

    for j in range(num_joints):
        vis = joints_vis[j]

        # if joint has low visible value, skip
        if vis < 0.5:
            continue

        target_weight[j, 0] = 1.0
        x, y = joints[j]

        # set the center pixel of Gaussian Kernel in entire window
        center_x = int(x + 0.5)
        center_y = int(y + 0.5)

        # square area that going to express Gaussian heatmap
        x0 = int(center_x - kernel_radius)
        y0 = int(center_y - kernel_radius)
        x1 = int(center_x + kernel_radius + 1)
        y1 = int(center_y + kernel_radius + 1)

        # upper left
        ul = [x0, y0]

        # bottom right
        br = [x1, y1]

        # skip if out of area
        if (ul[0] >= W) or (ul[1] >= H) or (br[0] < 0) or (br[1] < 0):
            continue

        kernel_size = int(2 * kernel_radius + 1)

        # (x,y) grid for the Gaussian kernel
        x_grid = np.arange(kernel_size, dtype=np.float32)
        y_grid = x_grid[:, None]

        # center pixel in Gaussian heatmap
        center_x_in_kernel = center_x - x0
        center_y_in_kernel = center_y - y0
        gauss = np.exp(-(((x_grid - center_x_in_kernel)**2 +
                          (y_grid - center_y_in_kernel)**2) / (2 * sigma**2)))

        # image slice
        # valid area on the heatmap where the kernel can be applied
        x0_img, x1_img = max(0, x0), min(x1, W)
        y0_img, y1_img = max(0, y0), min(y1, H)

        # kernel slice
        # crop the kernel to match the valid area on the heatmap
        x0_ker, x1_ker = x0_img - x0, x1_img - x0
        y0_ker, y1_ker = y0_img - y0, y1_img - y0

        # 1. Select the region on the heatmap where the Gaussian will be applied
        # 2. Select the corresponding region from the Gaussian kernel
        # 3. Take the pixel maximum to preserve the strongest values
        # 4. Write result back to the heatmap
        target[j, y0_img:y1_img, x0_img:x1_img] = np.maximum(
            target[j, y0_img:y1_img, x0_img:x1_img],
            gauss[y0_ker:y1_ker, x0_ker:x1_ker]
        )
    return target, target_weight


# Make the bbox match the input aspect ratio to avoid shape distortion.
def match_bbox(bbox, img_size, ratio=RATIO, scale=BBOX_SCALE):

    # original coordinates of image
    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    # box center
    cx = x + w / 2.0
    cy = y + h / 2.0

    # scale
    w *= scale
    h *= scale

    # If bbox is wider than target extend height. else, extend width.
    cur_ratio = w / h
    if cur_ratio > ratio:
        h = w / ratio
    else:
        w = h * ratio

    # calculate coordinate again
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0

    W_img, H_img = img_size
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    w = min(w, W_img - x0)
    h = min(h, H_img - y0)

    # make at least 2 pixel for w,h
    w = max(2.0, w)
    h = max(2.0, h)

    # these return values will go into crop_and_resize to cropped and resized
    return [x0, y0, w, h]


def crop_and_resize(src_img, bbox, input_size=(IN_W, IN_H)):

    # Unpack bbox [x, y, w, h]
    x_min, y_min, box_w, box_h = bbox

    # Original image size
    img_w, img_h = src_img.size

    # Fix bbox coordinates to lie inside the image
    x0 = max(0, int(x_min))
    y0 = max(0, int(y_min))
    x1 = min(img_w, int(x_min + box_w))
    y1 = min(img_h, int(y_min + box_h))

    # If bbox is invalid, we will use entire image
    if x1 <= x0 or y1 <= y0:
        crop_img = src_img
        crop_x0, crop_y0 = 0, 0
    else:
        # If valid crop the person region
        crop_img = src_img.crop((x0, y0, x1, y1))
        crop_x0, crop_y0 = x0, y0

    # Size of the cropped region
    crop_w, crop_h = crop_img.size

    # Model input size
    out_w, out_h = input_size

    # Resize the cropped match to the model input size and interpolation with Bilinear
    resized = crop_img.resize((out_w, out_h), Image.BILINEAR)

    # scale factor
    scale_x = out_w / max(1, crop_w)
    scale_y = out_h / max(1, crop_h)

    # Return the resized image and a tuple
    return resized, (crop_x0, crop_y0, scale_x, scale_y)


def transform_keypoints_to_input(keypoints_xy, crop_to_input):
    # keypoints_xy: original image of x,y
    # crop_to_input: return value from crop_and_resize
    crop_x0, crop_y0, scale_x, scale_y = crop_to_input

    kp_input = keypoints_xy.copy()

    # translate from original → crop coords, then scale to input size
    kp_input[:, 0] = (kp_input[:, 0] - crop_x0) * scale_x
    kp_input[:, 1] = (kp_input[:, 1] - crop_y0) * scale_y

    # keypoints in resized image
    return kp_input


"""
Our COCO lower-body dataset class is adapted from Microsoft’s SimpleBaseline
implementation. We follow their top-down COCO pipeline, but modify it for
six lower-body joints and a simplified crop-and-resize step.
"""
class COCOKeypointsLowerGTBbox(Dataset):
    def __init__(self, img_dir, ann_file, is_train=True, augment=False):

        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.is_train = is_train
        self.augment = augment

        # Build a list of (image_id, annotation_id) pairs.
        # Each person with valid keypoints becomes one sample.
        self.ids = []
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            # Get person annotations for image
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False, catIds=[1])
            anns = self.coco.loadAnns(ann_ids)
            for a in anns:
                # Skip instances without keypoints
                if a.get('num_keypoints', 0) <= 0:
                    continue
                self.ids.append((img_id, a['id']))

    # Number of (image, person) pairs
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image and annotation IDs for sample
        img_id, ann_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        ann = self.coco.loadAnns([ann_id])[0]

        # load image
        path = os.path.join(self.img_dir, img_info['file_name'])
        src_img = Image.open(path).convert('RGB')

        # COCO keypoints: (17, 3) = [x, y, visibility]
        kp_full = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        joints_full_xy = kp_full[:, :2]
        vis_full = (kp_full[:, 2] > 0).astype(np.float32)

        # Use only lower-body joints (6 joints)
        joints_xy = joints_full_xy[LOWER_IDX].copy()
        joints_vis = vis_full[LOWER_IDX].copy()

        # Person bbox in original image coords [x, y, w, h]
        bbox_person = ann['bbox']
        bbox = match_bbox(
            bbox_person,
            (src_img.size[0], src_img.size[1]),  # (W_img, H_img)
            ratio=RATIO,
            scale=BBOX_SCALE
        )

        # crop and resize to input model size
        cropped_img, crop_to_input = crop_and_resize(src_img, bbox, (IN_W, IN_H))

        # transform keypoints to match input size
        kpts_input = transform_keypoints_to_input(joints_xy, crop_to_input)

        # Horizontal flip augmentation
        if self.is_train:
            if self.augment and random.random() < 0.5:
                # Flip image horizontally
                cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                # Flip x-coordinates of keypoints
                kpts_input[:, 0] = IN_W - 1 - kpts_input[:, 0]
                # Swap left/right joint indices
                for a, b in FLIP_PAIRS:
                    kpts_input[[a, b]] = kpts_input[[b, a]]
                    joints_vis[[a, b]] = joints_vis[[b, a]]

        # normalize by ImageNet
        img_np = np.array(cropped_img, dtype=np.float32) / 255.0
        img_chw = img_np.transpose(2, 0, 1)
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        img_chw = (img_chw - IMAGENET_MEAN) / IMAGENET_STD

        # Downscale keypoints to heatmap size
        kpts_hm = kpts_input / 4.0
        target, target_weight = generate_target_heatmaps(
            kpts_hm, joints_vis, num_joints=NUM_JOINTS, hm_size=(HM_W, HM_H), sigma=SIGMA
        )

        # Pack everything into a sample dict for the DataLoader
        sample = {
            'image': torch.from_numpy(img_chw),
            'target': torch.from_numpy(target),
            'target_weight': torch.from_numpy(target_weight),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'image_id': img_id,
            'ann_id': ann_id,
            'kpts_in': torch.from_numpy(kpts_input),
            'joints_vis': torch.from_numpy(joints_vis),
        }
        return sample


# Heatmap decoder
def decode_heatmaps(hm, beta=100.0, stride=4.0):
    B, J, H, W = hm.shape
    hm_flat = hm.reshape(B, J, -1)

    hm_max = hm_flat.max(dim=-1, keepdim=True).values
    prob = torch.softmax(beta * (hm_flat - hm_max), dim=-1)   # (B,J,H*W)

    xs = torch.arange(W, device=hm.device, dtype=hm.dtype)
    ys = torch.arange(H, device=hm.device, dtype=hm.dtype)
    xs_grid = xs.unsqueeze(0).repeat(H, 1)          # (H,W)
    ys_grid = ys.unsqueeze(1).repeat(1, W)          # (H,W)
    grid = torch.stack([xs_grid, ys_grid], dim=-1).view(-1, 2)  # (H*W,2)

    exp_xy = torch.matmul(prob, grid)
    xs_in = exp_xy[..., 0] * stride
    ys_in = exp_xy[..., 1] * stride

    conf = prob.max(dim=-1).values
    return xs_in, ys_in, conf


# Make model and loss
from lib.models.pose_resnet18 import PoseResNet18
from lib.models.pose_resnet18_fpn import PoseResNet18FPN
from lib.models.pose_lower_cnn import LowerBodyPoseNet

def build_model():
    return PoseResNet18(num_joints=NUM_JOINTS)


"""
The JointsMSELoss adapted from Microsoft’s SimpleBaseline implementation.
which uses per-joint MSE over heatmaps.
"""
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        if self.use_target_weight:
            B, J, H, W = output.shape
            output = output.view(B, J, -1)
            target = target.view(B, J, -1)
            loss = 0
            tw = target_weight
            for j in range(J):
                loss += self.criterion(output[:, j, :] * tw[:, j, :],
                                       target[:, j, :] * tw[:, j, :])
            return loss / J
        else:
            return self.criterion(output, target)


def pck_accuracy(pred_x, pred_y, gt_kpts_in, vis, thr):

    # Btach size and predicted joints count
    B, pred_joints = pred_x.shape
    # ground truth joints count
    gt_joints = gt_kpts_in.shape[1]

    # choose smaller for safe return value
    joints = min(pred_joints, gt_joints)
    pred_x = pred_x[:, :joints]
    pred_y = pred_y[:, :joints]
    gt_x = gt_kpts_in[:, :joints, 0]
    gt_y = gt_kpts_in[:, :joints, 1]
    vis_j = vis[:, :joints]

    # use joint only visible and inside of image
    valid = (
        (vis_j > 0.5) &
        (gt_x >= 0) & (gt_x < IN_W) &
        (gt_y >= 0) & (gt_y < IN_H)
    )

    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return 0.0

    # calculate distance
    dx = pred_x - gt_x
    dy = pred_y - gt_y
    dist = torch.sqrt(dx * dx + dy * dy)

    # normalize to image size
    norm = float(max(IN_W, IN_H))
    dist_norm = dist / norm

    # only under threshold and valid
    ok = (dist_norm <= thr) & valid

    pck = float(ok.sum().item() / num_valid)
    return pck


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_pck005 = 0.0
    total_pck01 = 0.0
    n_batches = 0

    # Loop over mini-batches from the DataLoader with a progress bar
    for batch in tqdm(loader, ncols=100):
        imgs = batch['image'].to(DEVICE).float()
        target = batch['target'].to(DEVICE).float()
        tw = batch['target_weight'].to(DEVICE).float()
        gt_in = batch['kpts_in'].to(DEVICE).float()
        vis = batch['joints_vis'].to(DEVICE).float()

        optimizer.zero_grad()

        # forward + loss
        out = model(imgs)
        loss = criterion(out, target, tw)

        # backward + update
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            px, py, conf = decode_heatmaps(out.detach())
            pck_005 = pck_accuracy(px, py, gt_in, vis, thr=0.05)
            pck_01 = pck_accuracy(px, py, gt_in, vis, thr=0.10)

        total_loss += loss.item()
        total_pck005 += pck_005
        total_pck01 += pck_01
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_pck005 = total_pck005 / n_batches
    avg_pck01 = total_pck01 / n_batches
    return avg_loss, avg_pck005, avg_pck01


@torch.no_grad()
def eval(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_pck005 = 0.0
    total_pck01 = 0.0
    n_batches = 0

    for batch in tqdm(loader, ncols=100):
        imgs = batch['image'].to(DEVICE).float()
        target = batch['target'].to(DEVICE).float()
        tw = batch['target_weight'].to(DEVICE).float()
        gt_in = batch['kpts_in'].to(DEVICE).float()
        vis = batch['joints_vis'].to(DEVICE).float()

        out = model(imgs)
        loss = criterion(out, target, tw)

        px, py, conf = decode_heatmaps(out)
        pck_005 = pck_accuracy(px, py, gt_in, vis, thr=0.05)
        pck_01 = pck_accuracy(px, py, gt_in, vis, thr=0.10)

        total_loss += loss.item()
        total_pck005 += pck_005
        total_pck01 += pck_01
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_pck005 = total_pck005 / n_batches
    avg_pck01 = total_pck01 / n_batches
    return avg_loss, avg_pck005, avg_pck01


# to make graph of training history
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_pck005': [],
    'val_pck005': [],
    'train_pck01': [],
    'val_pck01': [],
}


def main():

    # Build COCO lower-body datasets for train and validation
    train_set = COCOKeypointsLowerGTBbox(
        TRAIN_IMG_DIR,
        TRAIN_ANN,
        is_train=True,
        augment=True
    )
    val_set = COCOKeypointsLowerGTBbox(
        VAL_IMG_DIR,
        VAL_ANN,
        is_train=False,
        augment=False
    )

    # Wrap datasets with DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Build pose model and move it to 'cuda'
    model = build_model().to(DEVICE)
    criterion = JointsMSELoss(use_target_weight=True)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_pck005 = 0.0

    # Training loop over epochs
    for epoch in range(1, EPOCHS + 1):
        print(f'\n[Epoch {epoch}/{EPOCHS}]')
        tr_loss, tr_pck005, tr_pck01 = train(model, train_loader, criterion, optimizer)
        vl_loss, vl_pck005, vl_pck01 = eval(model, val_loader, criterion)

        print(f'train: loss={tr_loss:.4f}  pck@0.05={tr_pck005:.3f}  pck@0.10={tr_pck01:.3f}')
        print(f'valid: loss={vl_loss:.4f}  pck@0.05={vl_pck005:.3f}  pck@0.10={vl_pck01:.3f}')

        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_pck005'].append(tr_pck005)
        history['val_pck005'].append(vl_pck005)
        history['train_pck01'].append(tr_pck01)
        history['val_pck01'].append(vl_pck01)

        # Prepare checkpoint dict
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pck005': best_pck005,
            'run_tag': RUN_TAG
        }

        # Make sure output directory exists
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, OUT_DIR / f'last_{RUN_TAG}.pth')

        # If validation pck0.05 improved, update best checkpoint
        if vl_pck005 > best_pck005:
            best_pck005 = vl_pck005
            torch.save(ckpt, OUT_DIR / f'best_{RUN_TAG}.pth')
            print('best updated')

    plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='train loss')
    plt.plot(history['epoch'], history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Loss curve ({RUN_TAG})')
    plt.legend()
    plt.tight_layout()
    loss_fig_path = OUT_DIR / f'loss_curve_{RUN_TAG}.png'
    plt.savefig(loss_fig_path)
    plt.show()

    plt.figure()
    plt.plot(history['epoch'], history['train_pck005'], label='train pck@0.05')
    plt.plot(history['epoch'], history['val_pck005'], label='val pck@0.05')
    plt.plot(history['epoch'], history['train_pck01'], label='train pck@0.10')
    plt.plot(history['epoch'], history['val_pck01'], label='val pck@0.10')
    plt.xlabel('epoch')
    plt.ylabel('PCK')
    plt.title(f'PCK curves ({RUN_TAG})')
    plt.legend()
    plt.tight_layout()
    pck_fig_path = OUT_DIR / f'pck_curve_{RUN_TAG}.png'
    plt.savefig(pck_fig_path)
    plt.show()

if __name__ == '__main__':
    main()
