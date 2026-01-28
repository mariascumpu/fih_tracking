from ultralytics import YOLO
from pathlib import Path
import os

DATA_YAML = Path(os.path.expanduser("~/Desktop/Maria_Master/dataset/data.yaml"))

def main():
    # 1) Start from a pretrained base model
    model = YOLO("yolov8n.pt")

    # 2) Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=16,
        patience=20,
        project="runs",          # will create ./runs/detect/...
        name="fish_yolov8n_v1",
        device="cpu",
    )

    # 3) Load best weights from THIS run
    best_path = results.save_dir / "weights" / "best.pt"
    print("Best weights:", best_path)

    best = YOLO(str(best_path))

    # 4) Validate using best weights
    val_results = best.val(data=str(DATA_YAML), imgsz=640)
    print(val_results)

    print("Training finished.")

    # OPTIONAL: tracking (video)
    # best.track(
    #     source="/path/to/video.mp4",
    #     tracker="bytetrack.yaml",
    #     imgsz=640,
    #     conf=0.25,
    #     save=True
    # )



if __name__ == "__main__":
    main()
