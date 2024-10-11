import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("../yolov8n.pt")

# 初始化目标跟踪器
tracker = sv.ByteTrack()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    # 获得Detections结果
    detections = sv.Detections.from_ultralytics(results)
    # 轨迹跟踪
    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


sv.process_video(
    source_path="https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
    # 输出结果参考：https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4
    target_path="../output.mp4",
    callback=callback
)