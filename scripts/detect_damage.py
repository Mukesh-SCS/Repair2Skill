"""
================================================================================
Detect Damage and Part Type using Faster R-CNN
================================================================================
Stage 1–2: detect parts & damages, pair them, then generate plan + graph.
Writes:
  outputs/stage1_parts.json
  outputs/repair_plan_<part>_<damage>.json
  outputs/repair_graph_<part>.json
  data/visual_guides/<part>_repair_guide.png

Usage:
  from scripts.detect_damage import detect_damage_and_parts
  detect_damage_and_parts("chair.jpg", "./models/damage_detection/frcnn_model.pth")
================================================================================
"""

import os, json, torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# keep order: parts then damages
PART_CLASSES   = ["seat","back","front_left_leg","front_right_leg","back_left_leg","back_right_leg","armrest_left","armrest_right"]
DAMAGE_CLASSES = ["missing","cracked","broken","loose","scratched"]
ALL_CLASSES    = PART_CLASSES + DAMAGE_CLASSES   # background is implicit (+1)

def _build_model():
    m = fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, 1 + len(ALL_CLASSES))
    return m

@torch.no_grad()
def detect_damage_and_parts(image_path: str, weights: str, threshold: float = 0.25, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    out = model(x)[0]
    boxes, scores, labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()

    parts, damages = [], []
    for b, s, l in zip(boxes, scores, labels):
        if s < threshold or s < 0.5:
            continue
        idx = int(l.item()) - 1                       # shift for background
        if idx < 0 or idx >= len(ALL_CLASSES): 
            continue
        name = ALL_CLASSES[idx]
        rec = {"box": b.tolist(), "confidence": float(s)}
        if name in PART_CLASSES:
            rec["part"] = name; parts.append(rec)
        else:
            rec["type"] = name; damages.append(rec)

    # Pair by IoU; if no IoU>0.3, pair by nearest centers as fallback
    pairs = []
    if parts and damages:
        pb = torch.tensor([p["box"] for p in parts])
        db = torch.tensor([d["box"] for d in damages])
        ious = box_iou(pb, db) if len(parts) and len(damages) else torch.zeros((0,0))
        for i, p in enumerate(parts):
            if ious.numel() and torch.max(ious[i]) > 0.30:
                j = int(torch.argmax(ious[i]))
            else:
                # center-distance fallback
                px1, py1, px2, py2 = pb[i]
                pcx, pcy = (px1+px2)/2, (py1+py2)/2
                d_centers = []
                for j_, d in enumerate(damages):
                    dx1, dy1, dx2, dy2 = db[j_]
                    dcx, dcy = (dx1+dx2)/2, (dy1+dy2)/2
                    d_centers.append(((pcx-dcx).pow(2)+(pcy-dcy).pow(2)).sqrt().item())
                j = int(torch.tensor(d_centers).argmin())

            pairs.append({
                "part": parts[i]["part"],
                "damage_type": damages[j]["type"],
                "part_confidence": parts[i]["confidence"],
                "damage_confidence": damages[j]["confidence"]
            })

    res = {"detected_damages": damages, "detected_parts": parts, "detected_pairs": pairs}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage1_parts.json","w") as f: json.dump(res, f, indent=2)
    return res

def _generate_outputs(pairs):
    from scripts.generate_repair_plan import generate_repair_plan
    from scripts.repair_graph import generate_repair_graph, save_repair_graph_json, visualize_repair_graph
    from scripts.render_visual_guidance import render_step_visual

    for dp in pairs:
        part = dp["part"]; dmg = dp["damage_type"]

        # plan
        plan = generate_repair_plan("Chair", part, dmg)
        plan_path = f"outputs/repair_plan_{part}_{dmg}.json"
        with open(plan_path,"w") as f: json.dump(plan, f, indent=2)
        print(f"[OK] {plan_path}")

        # graph
        graph = generate_repair_graph(part)
        graph_json = save_repair_graph_json(graph, part)  # writes outputs/repair_graph_<part>.json
        visualize_repair_graph(graph, f"outputs/repair_graph_{part}.png")

        # visual guide
        os.makedirs("data/visual_guides", exist_ok=True)
        render_step_visual(
            model_path=None,
            highlighted_part_idx=PART_CLASSES.index(part),
            save_path=f"data/visual_guides/{part}_repair_guide.png",
            damage_report_path="outputs/stage1_parts.json",
            plan_json_path=plan_path
        )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="./models/damage_detection/frcnn_model.pth")
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    result = detect_damage_and_parts(args.image, args.weights, args.threshold)
    if not result["detected_pairs"]:
        print("[WARN] No confident part-damage pair found.")
        return
    _generate_outputs(result["detected_pairs"])

if __name__ == "__main__":
    main()
