from ultralytics import YOLO
import cv2

# Load YOLO model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")
#model = YOLO("yolov8s.pt")   # small
#model = YOLO("yolov8m.pt") # medium
#model = YOLO("yolov8l.pt") # large
#model = YOLO("yolov8x.pt") # extra large
#model = YOLO("yolov8n-seg.pt") #segmentation (detect + outline objects)
#model = YOLO("yolov8n-pose.pt")  #human pose estimation (keypoints like hands, legs, joints)
#model = YOLO("yolov8n-cls.pt") #classification

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
