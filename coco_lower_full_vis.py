import json
from pathlib import Path

# Path to original COCO keypoints JSON (train)
src_json = Path("data/coco/annotations/annotations/person_keypoints_train2017.json")

# Load the original COCO annotation file
with open(src_json, "r") as f:
    anno = json.load(f)

images = anno["images"]
anns = anno["annotations"]
cats = anno["categories"]

# Lower-body joint indices
LOWER_IDXS = [11, 12, 13, 14, 15, 16]


def lower_body_fully_labeled(keypoints):
    """
    Check if all lower-body joints are labeled.
    """
    for j in LOWER_IDXS:
        # visibility value for joint j
        v = keypoints[3*j + 2]
        # if not labeled at all skip
        if v == 0:
            return False
    return True

selected_anns = []
selected_img_ids = set()

for a in anns:
    kpts = a.get("keypoints", [])
    # skip if no keypoints or invalid length
    if not kpts or len(kpts) != 17 * 3:
        continue
    if lower_body_fully_labeled(kpts):
        selected_anns.append(a)
        selected_img_ids.add(a["image_id"])

selected_images = [img for img in images if img["id"] in selected_img_ids]

out_dir = Path("data/coco_lower_full")
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "person_keypoints_train2017_lower_full.json"

subset_anno = {
    "images": selected_images,
    "annotations": selected_anns,
    "categories": cats,
}

with open(out_json, "w") as f:
    json.dump(subset_anno, f)