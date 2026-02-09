import cv2
from facedetection import detect_faces

cap = cv2.VideoCapture(0)

def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(frame, (x1-100, y1), (x2, y2+100), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
    detections = detect_faces(frame_small)

    scaled_boxes = []
    for (x1, y1, x2, y2, score) in detections:
        scaled_boxes.append((
            int(x1 * 2),
            int(y1 * 2),
            int(x2 * 2),
            int(y2 * 2),
            score
        ))

    draw_boxes(frame, scaled_boxes)

    cv2.imshow("Face Detector (From Scratch)", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

