from HOG  import hog_descriptor
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2
WINDOW_SIZE = (32, 32)


def random_patch(img, size=(64, 64)):
    h, w = img.shape
    if h < size[0] or w < size[1]:
        return None
    y = np.random.randint(0, h - size[0])
    x = np.random.randint(0, w - size[1])
    return img[y:y+size[0], x:x+size[1]]


lfw = fetch_lfw_people(
    min_faces_per_person=500,  # mínimo número de imágenes por persona
    resize=0.5,               # reducir tamaño para acelerar
    color=False               # escala de grises
)

svm = LinearSVC(C=0.01)
trained = False
def train():
    global svm

    X = []  # features
    y = []  # labels

    for img in lfw.images:
        for _ in range(3):  # 3 negativos por imagen
            patch = random_patch(img, WINDOW_SIZE)
            if patch is None:
                continue
            hog = hog_descriptor(patch)
            X.append(hog)
            y.append(0)

    for img in lfw.images:
           img = cv2.resize(img, WINDOW_SIZE)
           hog = hog_descriptor(img)
           X.append(hog)
           y.append(1)


    return X,y

def get_svm():
    if not trained:
        X,y=train()
        svm.fit(X, y)
    return svm




