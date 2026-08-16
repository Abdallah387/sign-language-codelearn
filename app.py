"""
Live Sign Language Recognition + Word Builder

Controls:
    S          -> Add current predicted letter
    SPACE      -> Add space
    BACKSPACE  -> Delete last character
    C          -> Clear the whole word
    ENTER      -> Confirm word
    Q          -> Quit

Confidence threshold:
    50%

IMPORTANT: Click on the "Sign Language Recognition" video window before
pressing any key. If the window isn't focused, OpenCV never receives your
keystrokes. Also make sure your keyboard input language is set to English
while using this app -- if it's set to Arabic, the physical "S" key sends a
different character code and the S/C/Q shortcuts will silently do nothing.
"""

import os
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\project\project_codolearn_final\models\knn_model.joblib"

LABEL_MAPPING_PATH = r"D:\project\project_codolearn_final\label_mapping.json"

# Minimum confidence required to save a letter
CONFIDENCE_THRESHOLD = 0.50

# Set to True to print the raw key code of every key you press.
# Use this if S/C/Q still don't work -- it tells you exactly what code
# your keyboard is actually sending, so we can match against it.
DEBUG_KEYS = True


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_landmarks(xs, ys):
    """
    Same normalization used during training.

    Wrist-centered translation
    +
    scale using wrist -> middle MCP distance
    """

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Wrist = landmark 0
    wrist_x = xs[0]
    wrist_y = ys[0]

    # Move wrist to (0, 0)
    xs = xs - wrist_x
    ys = ys - wrist_y

    # Landmark 9 = Middle Finger MCP
    scale = np.sqrt(xs[9] ** 2 + ys[9] ** 2)

    if scale == 0:
        scale = 1e-6

    xs = xs / scale
    ys = ys / scale

    return xs, ys


# ============================================================
# BUILD FEATURES
# ============================================================

def build_feature_row(hand_landmarks):
    """
    MediaPipe hand landmarks
    ->
    42 features:

    x0, y0,
    x1, y1,
    ...
    x20, y20
    """

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    xs, ys = normalize_landmarks(xs, ys)

    row = {}

    for i in range(21):
        row[f"x{i}"] = xs[i]
        row[f"y{i}"] = ys[i]

    return pd.DataFrame([row])


# ============================================================
# KEY MATCHING HELPER
# ============================================================

def key_is(key, *chars):
    """
    Checks a waitKey() result against one or more target characters,
    matching case-insensitively regardless of which case ord() was given.
    """
    if key == -1 or key == 255:
        return False
    for ch in chars:
        if key in (ord(ch.lower()), ord(ch.upper())):
            return True
    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SIGN LANGUAGE RECOGNITION")
    print("=" * 60)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    if not os.path.exists(MODEL_PATH):

        print("ERROR: Model file not found!")
        print(MODEL_PATH)
        return

    try:

        model = joblib.load(MODEL_PATH)

    except Exception as e:

        print("ERROR loading model:")
        print(e)
        return

    print("Model loaded successfully.")

    # ========================================================
    # LOAD LABEL MAPPING
    # ========================================================

    if not os.path.exists(LABEL_MAPPING_PATH):

        print("ERROR: Label mapping file not found!")
        print(LABEL_MAPPING_PATH)
        return

    try:

        with open(
            LABEL_MAPPING_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            label_mapping = json.load(f)

    except Exception as e:

        print("ERROR loading label mapping:")
        print(e)
        return

    inv_label_mapping = {
        int(value): str(key)
        for key, value in label_mapping.items()
    }

    print("Label mapping loaded successfully.")

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("\nERROR: Could not open webcam.")
        print("Check camera permissions or camera index.")
        return

    # ========================================================
    # VARIABLES
    # ========================================================

    current_label = ""
    current_confidence = 0.0
    word = ""

    # ========================================================
    # MEDIAPIPE HANDS
    # ========================================================

    with mp_hands.Hands(

        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5

    ) as hands:

        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)
        print("S          -> Add current letter")
        print("SPACE      -> Add space")
        print("BACKSPACE  -> Delete last character")
        print("C          -> Clear word")
        print("ENTER      -> Confirm word")
        print("Q          -> Quit")
        print("=" * 60)
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")
        print("=" * 60)
        print()
        print("Click on the video window before pressing any key!")
        print()

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            success, frame = cap.read()

            if not success:
                print("Failed to read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # =================================================
            # HAND DETECTED
            # =================================================

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                try:
                    features = build_feature_row(hand_landmarks)
                    prediction = model.predict(features)[0]
                    prediction = int(prediction)

                    if prediction in inv_label_mapping:
                        current_label = inv_label_mapping[prediction]
                    else:
                        current_label = str(prediction)

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)
                        current_confidence = float(probabilities.max())
                    else:
                        current_confidence = 1.0

                except Exception as e:
                    print("Prediction error:", e)
                    current_label = ""
                    current_confidence = 0.0

            else:
                current_label = ""
                current_confidence = 0.0

            # =================================================
            # UI
            # =================================================

            height, width = frame.shape[:2]

            cv2.rectangle(frame, (0, 0), (width, 135), (0, 0, 0), -1)

            cv2.putText(
                frame, "Current:", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # -------------------------------------------------
            # CURRENT LABEL -- confidence percentage removed,
            # still color-coded green/red based on the threshold.
            # -------------------------------------------------
            if current_label:

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    text_color = (0, 255, 0)
                else:
                    text_color = (0, 0, 255)

                cv2.putText(
                    frame, current_label, (140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3
                )

            else:
                cv2.putText(
                    frame, "No hand", (140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.putText(
                frame, "Min: 50%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
            )

            cv2.putText(
                frame, "Word:", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            display_word = word
            if len(display_word) > 30:
                display_word = "..." + display_word[-30:]

            cv2.putText(
                frame, display_word, (110, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3
            )

            cv2.rectangle(frame, (0, height - 55), (width, height), (0, 0, 0), -1)

            controls = (
                "S:Add | SPACE:Space | "
                "BACKSPACE:Delete | "
                "C:Clear | ENTER:Done | Q:Quit"
            )

            cv2.putText(
                frame, controls, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1
            )

            cv2.imshow("Sign Language Recognition", frame)

            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if DEBUG_KEYS and key != 255:
                shown = chr(key) if 32 <= key <= 126 else "?"
                print(f"[debug] key code pressed: {key}  (char: {shown})")

            if key_is(key, "q"):
                break

            elif key_is(key, "s"):

                if current_label == "":
                    print("No letter detected.")
                    continue

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    word += current_label
                    print(f"Added: {current_label}")
                    print(f"Confidence: {current_confidence * 100:.1f}%")
                    print(f"Word: {word}")
                else:
                    print("Letter NOT added.")
                    print(f"Confidence: {current_confidence * 100:.1f}%")
                    print("Required: 50%")

            elif key == 32:
                if word and not word.endswith(" "):
                    word += " "
                    print("Space added.")
                    print(f"Word: {word}")

            elif key == 8:
                if word:
                    deleted = word[-1]
                    word = word[:-1]
                    print(f"Deleted: {deleted}")
                    print(f"Word: {word}")

            elif key_is(key, "c"):
                word = ""
                print("Word cleared.")

            elif key == 13:
                print()
                print("=" * 60)
                print("FINAL WORD")
                print("=" * 60)
                print(word)
                print("=" * 60)
                print()

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("PROGRAM CLOSED")
    print("=" * 60)
    print("Final word:", word)


if __name__ == "__main__":
    main()