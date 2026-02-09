import cv2
from HOG  import hog_descriptor as compute_hog
import numpy as np

from svm_training import get_svm
svm = get_svm()

WINDOW_SIZE = (32, 32)
STEP = 12
SCALE = 3
THRESHOLD = 0.2

#reducing image scale
def image_pyramid(img, scale=1.25, min_size=(64, 64)):
    yield img
    while True:
        w = int(img.shape[1] / scale)
        h = int(img.shape[0] / scale)
        if w < min_size[0] or h < min_size[1]:
            break
        img = cv2.resize(img, (w, h))
        yield img

#sliding window over image
def sliding_window(img, step=8, window_size=(64, 64)):
    for y in range(0, img.shape[0] - window_size[1], step):
        for x in range(0, img.shape[1] - window_size[0], step):
            yield (x, y, img[y:y+window_size[1], x:x+window_size[0]])



# face will be detected multiple times, so its needed to keep only the higher box
def non_max_suppression(boxes, threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes.T

    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[order[1:]]
        order = order[1:][overlap < threshold]

    return boxes[keep]

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections=[]
    orig_h, orig_w = gray.shape
    for scaled_img in image_pyramid(gray, scale=SCALE, min_size=WINDOW_SIZE):
        scale_factor = min(orig_w,orig_h) / scaled_img.shape[1]

        for (x, y, window) in sliding_window(
            scaled_img,
            step=STEP,
            window_size=WINDOW_SIZE
        ):
            if window.shape != WINDOW_SIZE:
                continue

            hog = compute_hog(window)
            score = svm.decision_function([hog])[0]

            if score > THRESHOLD:
                x1 = int(x * scale_factor)
                y1 = int(y * scale_factor)
                x2 = int((x + WINDOW_SIZE[0]) * scale_factor)
                y2 = int((y + WINDOW_SIZE[1]) * scale_factor)

                detections.append((x1, y1, x2, y2, score))

    final_boxes = non_max_suppression(detections, threshold=0.3)
    return final_boxes
