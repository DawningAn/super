import supervision as sv
# 打印supervision的版本
print(sv.__version__)

import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("../yolov8n.pt")
image = cv2.imread('../images/1.jpg')
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

# 标注识别框
annotated_image = box_annotator.annotate(scene=image, detections=detections)
# 标注识别标签和置信度
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
cv2.imwrite('../2.jpeg', annotated_image)
