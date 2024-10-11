import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets

# 越线统计

# 如果不存在则下载视频
video_name = download_assets(VideoAssets.VEHICLES)
# 加载 YoloV8n 模型，如果不存在会自动下载
model = YOLO("yolov8n.pt")

# 预设界限
start = sv.Point(0, 400)
end = sv.Point(1280, 400)
# 初始预线检测器
line_zone = sv.LineZone(
    start=start,
    end=end
)
# 追踪器
tracker = sv.ByteTrack()
# 初始化展现对象
trace_annotator = sv.TraceAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=1
)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

# 读取视频
cap = cv2.VideoCapture(video_name)
# 初始化计时器
prev_tick = cv2.getTickCount()
while True:
    # 循环读取每一帧
    ret, frame = cap.read()
    if not ret:
        break
    # 由于原视频帧比较大，方便处理和后面展现，缩小一些
    frame = cv2.resize(frame, (1280, 720))
    result = model(
        frame,
        device=[0]  # 如果是 cpu 则是 device='cpu'
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    # 目标跟踪
    detections = tracker.update_with_detections(detections)
    # 更新预线检测器
    crossed_in, crossed_out = line_zone.trigger(detections)
    print(f'in:{line_zone.in_count}', f'out:{line_zone.out_count}')
    # 获得各边界框的标签
    labels = [
        f"{result.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    # 绘制轨迹
    frame = trace_annotator.annotate(frame, detections=detections)
    # 绘制标签
    frame = label_annotator.annotate(frame, detections=detections, labels=labels)
    # 绘制预制线
    frame = line_zone_annotator.annotate(frame, line_counter=line_zone)

    # 计算帧率
    current_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_tick - prev_tick)
    prev_tick = current_tick
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('video', frame)
    cv2.waitKey(1)

