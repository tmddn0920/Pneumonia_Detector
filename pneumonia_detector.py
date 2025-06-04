import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from segmentation import apply_lung_mask
from feature_extraction import extract_features

model = joblib.load("models/pneumonia_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def choose_image_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="X-ray 이미지 선택",
        filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp"))]
    )
    return file_path

def predict_pneumonia_from_path(image_path):
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, (256, 256))

    masked_img, mask = apply_lung_mask(original_img)
    features, processed_img = extract_features(masked_img, mask, label=None, return_processed=True)

    input_df = pd.DataFrame([features])
    for col in ["filename", "label"]:
        if col in input_df.columns:
            input_df = input_df.drop(columns=col)
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0][1]
    pred = 1 if proba >= 0.8 else 0

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Lung Mask")
    axs[2].imshow(processed_img, cmap='gray')
    axs[2].set_title("CLAHE Enhanced")
    for ax in axs:
        ax.axis('off')
    result_text = f"Prediction: {'PNEUMONIA' if pred == 1 else 'NORMAL'} (Probability: {proba:.2f})"
    plt.suptitle(result_text, fontsize=16, color='red' if pred == 1 else 'green')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = choose_image_path()
    if image_path:
        predict_pneumonia_from_path(image_path)
    else:
        print("이미지를 선택하지 않았습니다.")
