from ultralytics import YOLO

model = YOLO("runs/detect/ppe_model4/weights/best.pt")

results = model.predict(
    source="C:/Users/yyash/OneDrive/Desktop/construction-ppe/images/test/real_image1.jpg",
    save=True
)

print("Detection completed!")
