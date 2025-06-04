import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy
from image_loader import load_images_from_folder
from segmentation import apply_lung_mask

def preprocess_masked(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply((img * 255).astype(np.uint8))
    return img.astype(np.float32) / 255.0

def extract_features(image, mask, label, filename="unknown", return_processed=False):
    masked = image * mask
    masked = preprocess_masked(masked)
    h, w = mask.shape

    mean_intensity = np.mean(masked[mask == 1]) if np.any(mask) else 0
    std_intensity = np.std(masked[mask == 1]) if np.any(mask) else 0
    area_ratio = np.sum(mask) / (h * w)

    edge_img = cv2.Canny((masked * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edge_img > 0) / np.sum(mask) if np.sum(mask) > 0 else 0

    glcm = graycomatrix((masked * 255).astype(np.uint8), distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    left_mask = mask[:, :w//2]
    right_mask = mask[:, w//2:]
    left_img = masked[:, :w//2]
    right_img = masked[:, w//2:]

    left_mean = np.mean(left_img[left_mask == 1]) if np.any(left_mask) else 0
    right_mean = np.mean(right_img[right_mask == 1]) if np.any(right_mask) else 0
    brightness_diff = abs(left_mean - right_mean)

    left_edges = cv2.Canny((left_img * 255).astype(np.uint8), 50, 150)
    right_edges = cv2.Canny((right_img * 255).astype(np.uint8), 50, 150)
    left_edge_density = np.sum(left_edges > 0) / np.sum(left_mask) if np.sum(left_mask) > 0 else 0
    right_edge_density = np.sum(right_edges > 0) / np.sum(right_mask) if np.sum(right_mask) > 0 else 0
    edge_diff = abs(left_edge_density - right_edge_density)

    left_lbp = local_binary_pattern((left_img * 255).astype(np.uint8), P=8, R=1, method='uniform')
    right_lbp = local_binary_pattern((right_img * 255).astype(np.uint8), P=8, R=1, method='uniform')
    left_hist, _ = np.histogram(left_lbp.ravel(), bins=np.arange(0, 10), density=True)
    right_hist, _ = np.histogram(right_lbp.ravel(), bins=np.arange(0, 10), density=True)
    left_entropy = entropy(left_hist + 1e-7)
    right_entropy = entropy(right_hist + 1e-7)
    entropy_diff = abs(left_entropy - right_entropy)
    
    feature_dict = {
        "filename": filename,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "area_ratio": area_ratio,
        "edge_density": edge_density,
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "brightness_asymmetry": brightness_diff,
        "edge_density_asymmetry": edge_diff,
        "lbp_entropy_asymmetry": entropy_diff,
    }

    if label is not None:
        feature_dict["label"] = label

    if return_processed:
        return feature_dict, masked
    else:
        return feature_dict

def process_all_images(normal_path, pneumonia_path, output_csv="feature_data.csv"):
    data = []
    normal_images = load_images_from_folder(normal_path, 0)
    pneumonia_images = load_images_from_folder(pneumonia_path, 1)
    all_images = normal_images + pneumonia_images

    for img, label in all_images:
        masked_img, mask = apply_lung_mask(img)
        features = extract_features(img, mask, label)
        data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"특징 저장 완료: {output_csv}")
    return df