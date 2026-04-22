from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=500,
        batch=15,
        name="ppe_model"
    )

if __name__ == "__main__":
    main()