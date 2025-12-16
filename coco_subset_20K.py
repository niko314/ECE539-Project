import json
import random
from pathlib import Path

# Path to original COCO keypoints JSON (train)
src_json = Path("data/coco/annotations/annotations/person_keypoints_train2017.json")

# Load the original COCO annotation file
with open(src_json, "r") as f:
    anno = json.load(f)

images = anno["images"]
anns = anno["annotations"]
cats = anno["categories"]

# Collect only annotation has at least 1 labeled keypoint
valid_anns = []
for ann in anns:
    num_kpts = ann.get("num_keypoints", 0)
    # only if it has some keypoints
    if num_kpts > 0:
        valid_anns.append(ann)

random.seed(42)
random.shuffle(valid_anns)

# select 20,000 person instances
selected_anns = valid_anns[:20000]

selected_img_ids = {a["image_id"] for a in selected_anns}

# Filter images
selected_images = [
    img for img in images
    if img["id"] in selected_img_ids
]

subset_anno = {
    "images": selected_images,
    "annotations": selected_anns,
    "categories": cats,
}

out_dir = Path("data/coco_mini")
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "person_keypoints_train2017_20k.json"

with open(out_json, "w") as f:
    json.dump(subset_anno, f)
