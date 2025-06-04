import os
import cv2
import numpy as np

NORMAL_PATH = 'data/NORMAL'
PNEUMONIA_PATH = 'data/PNEUMONIA'
IMAGE_SIZE = (256, 256)

def load_images_from_folder(folder, label):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            images.append((img, label))
    return images

def load_all_data(normal_path=NORMAL_PATH, pneumonia_path=PNEUMONIA_PATH):
    normal_data = load_images_from_folder(normal_path, 0)
    pneumonia_data = load_images_from_folder(pneumonia_path, 1)

    all_data = normal_data + pneumonia_data
    np.random.shuffle(all_data)

    X = np.array([img for img, label in all_data])
    y = np.array([label for img, label in all_data])

    X = X / 255.0
    X = X.reshape(-1, 256, 256, 1)

    return X, y

if __name__ == "__main__":
    X, y = load_all_data()
    print(f"전체 이미지 수: {len(X)}개")
    print(f"NORMAL: {sum(y == 0)}개, PNEUMONIA: {sum(y == 1)}개")
