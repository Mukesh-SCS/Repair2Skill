"""
================================================================================
DESCRIPTION:
    Trains a Faster R-CNN detector on the synthetic annotations to localize
    damaged parts. Saves a PyTorch state_dict for inference.

USAGE:
    python scripts/train_detector_frcnn.py \
      --ann ./data/synthetic_damage/annotations.json \
      --imgs ./data/synthetic_damage/images \
      --epochs 10 --batch 2

OUTPUTS:
    ./models/damage_detection/frcnn_model.pth

ARGUMENTS:
    --ann     Path to annotations.json
    --imgs    Path to images folder
    --epochs  Number of epochs (default 10)
    --batch   Batch size (default 2)
Author Info: Mukesh Mani Tripathi
================================================================================
"""

import os, json, argparse, torch, torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class FurnitureDetectionDataset(Dataset):
    def __init__(self, ann, imgs, transforms=None, max_items=None):
        with open(ann, "r") as f:
            self.annotations = json.load(f)
        if max_items: self.annotations = self.annotations[:max_items]
        self.images_dir, self.transforms = imgs, transforms
        self.class_names = ["__background__","seat","back","front_left_leg","front_right_leg",
            "back_left_leg","back_right_leg","armrest_left","armrest_right",
            "missing","cracked","broken","loose","scratched"]
        self.name_to_idx = {n:i for i,n in enumerate(self.class_names)}
    def __len__(self): return len(self.annotations)
    def __getitem__(self, idx):
        a = self.annotations[idx]
        image = Image.open(os.path.join(self.images_dir, a["filename"])).convert("RGB")
        boxes, labels = [], []
        parts = a.get("parts", {})
        for d in a.get("damages", []):
            p, t = d["part"], d["type"]
            if p in parts:
                x1,y1,x2,y2 = parts[p]
                boxes.append([x1,y1,x2,y2]); labels.append(self.name_to_idx[t])
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx])}
        if self.transforms: image = self.transforms(image)
        return image, target

def get_transform(resize):
    tf = [T.ToTensor()]
    if resize: tf.insert(0, T.Resize(resize))  # shrink long side
    return T.Compose(tf)

def build_model(arch, num_classes):
    if arch == "mobilenet":  # much faster on CPU
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    # trim RPN to reduce compute
    model.rpn.pre_nms_top_n_training = 1000
    model.rpn.post_nms_top_n_training = 500
    model.rpn.pre_nms_top_n_test = 500
    model.rpn.post_nms_top_n_test = 300
    return model

def collate_fn(batch): return tuple(zip(*batch))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="./data/synthetic_damage/annotations.json")
    ap.add_argument("--imgs", default="./data/synthetic_damage/images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arch", choices=["resnet50","mobilenet"], default="resnet50")  # set to mobilenet for speed
    ap.add_argument("--resize", type=int, default=512)   # set 320–640. Lower == faster
    ap.add_argument("--max-images", type=int, default=200)  # limit dataset for quick runs
    ap.add_argument("--workers", type=int, default=0)    # 0 on Windows CPU is safer
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count()//2))  # reduce thread thrash on CPU

    dataset = FurnitureDetectionDataset(args.ann, args.imgs,
                                        transforms=get_transform(args.resize),
                                        max_items=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.workers,
                        pin_memory=False, persistent_workers=False)

    num_classes = 14
    model = build_model(args.arch, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # warm-up: freeze backbone for epoch 1 on CPU
    for p in model.backbone.parameters(): p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(args.epochs):
        if epoch == 1:
            for p in model.backbone.parameters(): p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {float(loss):.4f}")

    os.makedirs("./models/damage_detection", exist_ok=True)
    torch.save(model.state_dict(), "./models/damage_detection/frcnn_model.pth")
    print("Saved ./models/damage_detection/frcnn_model.pth")

if __name__ == "__main__":
    main()
