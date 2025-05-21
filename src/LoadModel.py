import os
import numpy as np
import tensorflow as tf
import cv2
import json
from tcn import TCN

# Load model and classes
model = tf.keras.models.load_model(
    "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_Optimized_full.h5",
    custom_objects={'TCN': TCN}
)

with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_FusionFull.json", 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Preprocessing function
def preprocess_patch(patch, img_size=(128, 128)):
    patch = cv2.resize(patch, img_size)
    patch = np.expand_dims(patch, axis=0) / 255.0
    return patch

# NMS function
def non_max_suppression_fast(boxes, scores, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    idxs = np.argsort(scores)[::-1]
    selected_idxs = []

    while len(idxs) > 0:
        i = idxs[0]
        selected_idxs.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / ((x2[i] - x1[i]) * (y2[i] - y1[i]))

        idxs = idxs[np.where(overlap <= overlapThresh)[0] + 1]

    return boxes[selected_idxs], selected_idxs

# Frame detection function
def detect_jets_in_frame(frame, patch_size=128, step_size=32, threshold=0.7):
    height, width, _ = frame.shape
    boxes, scores, classes = [], [], []

    heatmap_alien = np.zeros((height, width), dtype=np.float32)
    heatmap_friendly = np.zeros((height, width), dtype=np.float32)

    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            patch = frame[y:y + patch_size, x:x + patch_size]
            processed_patch = preprocess_patch(patch)
            predictions = model.predict(processed_patch, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]

            if confidence >= threshold:
                boxes.append([x, y, patch_size, patch_size])
                scores.append(confidence)
                classes.append(class_names[predicted_index])

                if class_names[predicted_index] == "Alien":
                    heatmap_alien[y:y + patch_size, x:x + patch_size] += confidence
                elif class_names[predicted_index] == "Friendly":
                    heatmap_friendly[y:y + patch_size, x:x + patch_size] += confidence

    final_boxes, selected_idxs = non_max_suppression_fast(boxes, scores)

    heatmap_alien = cv2.normalize(heatmap_alien, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_friendly = cv2.normalize(heatmap_friendly, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_alien = cv2.applyColorMap(heatmap_alien.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_friendly = cv2.applyColorMap(heatmap_friendly.astype(np.uint8), cv2.COLORMAP_SPRING)

    frame_with_heatmap = frame.copy()
    frame_with_heatmap = cv2.addWeighted(frame_with_heatmap, 1.0, heatmap_alien, 0.5, 0)
    frame_with_heatmap = cv2.addWeighted(frame_with_heatmap, 1.0, heatmap_friendly, 0.5, 0)

    alien_detected = False
    friendly_detected = False

    for i, idx in enumerate(selected_idxs):
        x, y, w, h = boxes[idx]
        cls = classes[idx]
        conf = scores[idx]

        if cls == "Alien" and not alien_detected:
            print(f"Class = Alien | Confidence = {conf:.2f} | Location = ({x}, {y})")
            cv2.rectangle(frame_with_heatmap, (x, y), (x + w, y + h), (0, 0, 255), 2)
            alien_detected = True

        elif cls == "Friendly" and not friendly_detected:
            print(f"Class = Friendly | Confidence = {conf:.2f} | Location = ({x}, {y})")
            cv2.rectangle(frame_with_heatmap, (x, y), (x + w, y + h), (0, 255, 0), 2)
            friendly_detected = True

        if alien_detected and friendly_detected:
            break

    return frame_with_heatmap

# ---------- MAIN IMAGE PROCESSING ----------
input_path = r"C:\Users\lohit ramaraju\Downloads\combined.png"  # Change this to directory path if needed
output_dir = r"C:/Users/lohit ramaraju/Downloads/processed_images"

os.makedirs(output_dir, exist_ok=True)

if os.path.isfile(input_path):
    # Single image file
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error reading image: {input_path}")
    else:
        print(f"\nProcessing Input...")
        result = detect_jets_in_frame(image)
        out_path = os.path.join(output_dir, f"processed_{os.path.basename(input_path)}")
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")

elif os.path.isdir(input_path):
    # Directory of images
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error reading image: {filename}")
                continue

            print(f"\nProcessing {filename}...")
            result = detect_jets_in_frame(image)

            out_path = os.path.join(output_dir, f"processed_{filename}")
            cv2.imwrite(out_path, result)
            print(f"Saved: {out_path}")
else:
    print(f"Invalid input path: {input_path}")
