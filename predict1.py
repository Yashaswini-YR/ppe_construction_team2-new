from ultralytics import YOLO

model = YOLO("runs/detect/ppe_model_1/weights/best.pt")

results = model.predict(
    source="C:/Users/yyash/OneDrive/Desktop/construction-ppe/images/test/image441.jpg",
    save=True
)

print("Detection completed!")
