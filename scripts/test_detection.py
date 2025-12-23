"""
Quick test script to verify damage detection is working.
Run this with: python scripts/test_detection.py --image path/to/image.jpg
"""
import sys
import json
from detect_damage import detect

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_detection.py <image_path> [--threshold 0.05]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[sys.argv.index('--threshold') + 1]) if '--threshold' in sys.argv else 0.05
    
    print(f"[TEST] Testing detection on: {image_path}")
    print(f"[TEST] Using threshold: {threshold}")
    print("-" * 60)
    
    result = detect(image_path, threshold=threshold, debug=True)
    
    print("-" * 60)
    print(f"[TEST] Detection Result:")
    print(json.dumps(result, indent=2))
    
    if result.get('detected_pairs'):
        print(f"\n[TEST] ✓ SUCCESS: Detected {len(result['detected_pairs'])} damage(s)")
        for pair in result['detected_pairs']:
            print(f"  - {pair['part']} is {pair['damage_type']} (confidence: {pair['damage_confidence']:.3f})")
    else:
        print(f"\n[TEST] ✗ FAILED: No damage detected")
        print("[HINT] Try:")
        print("  1. Lower the threshold: --threshold 0.03")
        print("  2. Use a different image with more obvious damage")
        print("  3. Retrain the model if it's not detecting anything")

