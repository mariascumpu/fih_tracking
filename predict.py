from ultralytics import YOLO
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("~/Desktop/Maria_Master/dataset/data.yaml")
        sys.exit(1)

    weights = sys.argv[1]
    source = sys.argv[2]

    model = YOLO(weights)
    results = model.predict(source=source, save=True, conf=0.15, iou=0.5)

    print("Done. Look in runs/detect/ for outputs.")
    print(f"Predicted on: {source}")
    print(f"Results objects: {len(results)}")

if __name__ == "__main__":
    main()