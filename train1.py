from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 model (small + fast)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="data.yaml",   # your dataset config file
        epochs=30,          # you can change (e.g., 20 for quick demo)
        imgsz=500,          # image size
        batch=15,           # reduce to 8 if system is slow
        name="ppe_model_1"  # folder name for results
    )

if __name__ == "__main__":
    main()