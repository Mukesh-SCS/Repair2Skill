"""
Pipeline: detection JSON → damage report (Step 1) → deterministic plan (Step 2) → validate (Step 3).
Outputs strict damage report and a valid plan (only ONE damaged_part; seat never remove/replace).
Use from server: pass --detection <path> --plan <out> --damage-report <out>.
Prints one JSON line with plan_path, damage_report_path, damaged_part, damage_type (for server).
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.repair_planner import build_damage_report, get_valid_plan
from utils.repair_schema import normalize_part_name


def main():
    ap = argparse.ArgumentParser(description="Build damage report + valid repair plan from detection JSON")
    ap.add_argument("--detection", required=True, help="Path to detection JSON (from detect_damage.py)")
    ap.add_argument("--plan", required=True, help="Path to write repair plan JSON")
    ap.add_argument("--damage-report", required=True, help="Path to write damage report JSON")
    args = ap.parse_args()

    detection_path = Path(args.detection)
    plan_path = Path(args.plan)
    damage_report_path = Path(args.damage_report)

    if not detection_path.is_file():
        out = {"error": f"Detection file not found: {detection_path}", "plan_path": None, "damage_report_path": None}
        print(json.dumps(out))
        sys.exit(1)

    try:
        with open(detection_path, "r", encoding="utf-8") as f:
            detection = json.load(f)
    except Exception as e:
        out = {"error": f"Failed to read detection: {e}", "plan_path": None, "damage_report_path": None}
        print(json.dumps(out))
        sys.exit(1)

    # Step 1 — Damage report (single source; normalize part names)
    report = build_damage_report(detection)
    damaged_part = (report.get("damaged_part") or "").strip()
    damage_type = (report.get("damage_type") or "none").strip()
    confidence = float(report.get("confidence", 0))

    if not damaged_part:
        out = {
            "error": "No valid damaged part from detection (all pairs failed normalization or only seat detected)",
            "plan_path": None,
            "damage_report_path": None,
        }
        print(json.dumps(out))
        sys.exit(1)

    try:
        damaged_part = normalize_part_name(damaged_part)
    except ValueError as e:
        out = {"error": str(e), "plan_path": None, "damage_report_path": None}
        print(json.dumps(out))
        sys.exit(1)

    # If seat is the only "damaged" part: allow inspect-only plan (no remove/replace)
    report["damaged_part"] = damaged_part
    report["damage_type"] = damage_type
    report["confidence"] = confidence

    # Step 2 & 3 — Deterministic plan and validate (no LLM plan passed)
    plan = get_valid_plan(damaged_part, damage_type, llm_plan=None)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    damage_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    with open(damage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out = {
        "plan_path": str(plan_path.resolve()),
        "damage_report_path": str(damage_report_path.resolve()),
        "damaged_part": damaged_part,
        "damage_type": damage_type,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
