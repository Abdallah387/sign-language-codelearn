"""
Script to extract Hand Landmarks (x, y only) using MediaPipe
from an image dataset organized like this:

dataset/
    0/
        img1.jpg
        img2.jpg
    1/
        ...
    a/
        ...
    b/
        ...
    ...

and save the result into a CSV file.

Usage:
    python extract_landmarks.py --data_dir "path/to/dataset" --output_csv "landmarks.csv"

Requirements:
    pip install mediapipe opencv-python pandas --break-system-packages
"""

import os
import csv
import argparse
import cv2
import mediapipe as mp


def build_header(num_landmarks=21):
    """Builds the CSV header row"""
    header = ["label", "filename"]
    for i in range(num_landmarks):
        header.append(f"x{i}")
        header.append(f"y{i}")
         # Optional: include z-coordinate if needed
    return header


def extract_landmarks_from_image(hands_detector, image_path):
    """
    Reads the image, runs mediapipe on it, and returns a list of
    x,y values for all 21 landmarks if a hand is detected,
    or None if no hand was found.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Take the first detected hand if there's more than one in the image
    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []
    for landmark in hand_landmarks.landmark:
        coords.append(landmark.x)
        coords.append(landmark.y)
        # coords.append(landmark.z)  # Optional: include z-coordinate if needed

    return coords


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks (x,y,z) and save them to CSV")
    parser.add_argument("--data_dir", required=True, help="Path to the main dataset folder")
    parser.add_argument("--output_csv", default="landmarks.csv", help="Path/name of the output CSV file")
    parser.add_argument("--static_image_mode", action="store_true", default=True,
                         help="Static image mode (enabled by default since we're working with images, not video)")
    args = parser.parse_args()

    mp_hands = mp.solutions.hands

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    total_images = 0
    saved_images = 0
    skipped_images = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2,
    ) as hands_detector, open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(build_header())

        # Each folder inside data_dir is a label (digit or letter)
        labels = sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        )

        for label in labels:
            label_dir = os.path.join(args.data_dir, label)
            image_files = [
                f for f in sorted(os.listdir(label_dir))
                if f.lower().endswith(valid_extensions)
            ]

            print(f"[+] Processing folder: {label}  ({len(image_files)} images)")

            for filename in image_files:
                total_images += 1
                image_path = os.path.join(label_dir, filename)

                coords = extract_landmarks_from_image(hands_detector, image_path)

                if coords is None:
                    skipped_images += 1
                    continue

                writer.writerow([label, filename] + coords)
                saved_images += 1

    print("\n===== Summary =====")
    print(f"Total images: {total_images}")
    print(f"Images with hand landmarks extracted successfully: {saved_images}")
    print(f"Images skipped (no hand detected): {skipped_images}")
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()