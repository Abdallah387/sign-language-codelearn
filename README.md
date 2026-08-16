# Sign Language CodeLearn

A real-time sign-language recognition system that detects hand landmarks from images or a webcam, classifies hand gestures with machine-learning models, and builds words from the predicted letters.

The project combines computer vision, MediaPipe hand tracking, feature normalization, classical machine learning, data augmentation, hyperparameter tuning, and an OpenCV-based live application.

> **Project status:** The training and preprocessing notebooks are included. The large CSV datasets and trained model artifacts are stored separately from GitHub. Before running the webcam application, download the required model files and update the local paths in `app.py`.

## What the Project Does

The system recognizes hand gestures that represent letters or digits. During live use, it reads frames from the webcam, detects one hand, extracts the 21 MediaPipe hand landmarks, normalizes the coordinates, predicts the current class, and displays the result on the video window.

The application can turn individual predictions into a word:

```text
Webcam frame
     |
     v
MediaPipe hand detection
     |
     v
21 hand landmarks
     |
     v
Normalization and 42 numerical features
     |
     v
Trained classifier
     |
     v
Predicted letter + confidence
     |
     v
Word-building controls
```

## Key Features

- Hand detection with MediaPipe Hands.
- Extraction of 21 hand landmarks.
- XY-only and XYZ-based preprocessing pipelines.
- Wrist-centered translation and scale normalization.
- Stratified train/validation/test splitting.
- Training-data augmentation through small landmark rotations.
- Comparison of KNN, Random Forest, SVM, and XGBoost.
- Hyperparameter tuning with `GridSearchCV`.
- Soft-voting ensemble of the four classifiers.
- Real-time webcam inference through OpenCV.
- Confidence thresholding to reduce accidental letter insertion.
- Keyboard controls for constructing and editing words.

## Data Representation

Each detected hand contains 21 landmarks. The XY pipeline stores two coordinates per landmark:

```text
x0, y0, x1, y1, ..., x20, y20
```

This creates 42 input features. The XYZ pipeline additionally keeps the depth coordinate when it is available. Both pipelines maintain a label column and can include the original filename for traceability.

The project normalizes the coordinates by translating the wrist to the origin and scaling the hand using the distance from the wrist to the middle-finger MCP landmark. This reduces the influence of image position and hand size.

## Dataset Preparation

`extract_data.py` expects images to be organized in class folders. Each folder name is treated as a label:

```text
data/
├── 0/
│   ├── image_001.jpg
│   └── image_002.jpg
├── 1/
│   └── image_003.jpg
├── A/
│   └── image_004.jpg
└── B/
    └── image_005.jpg
```

The script reads each image, detects the first hand with MediaPipe, extracts the landmark coordinates, and saves successful detections to a CSV file. Images in which no hand is detected are skipped and counted in the final summary.

Run the extractor with:

```bash
python extract_data.py --data_dir "path/to/image_dataset" --output_csv "landmarks.csv"
```

The output contains a `label`, a `filename`, and landmark coordinate columns such as `x0`, `y0`, `x1`, and `y1`.

## Preprocessing Pipeline

The preprocessing notebooks prepare the raw landmark CSV for model training:

1. Load the raw landmark data.
2. Remove the `z` columns for the XY-only experiment.
3. Check class counts and coordinate ranges.
4. Visualize a sample hand.
5. Check missing values and duplicates.
6. Normalize the landmark coordinates.
7. Encode class names into numeric labels.
8. Create stratified train, validation, and test splits.
9. Augment the training set with small rotations around the wrist.
10. Save the processed CSV files and label mappings.

The main preprocessing files are:

| File | Purpose |
|---|---|
| `preprocessing_1.ipynb` | Prepares the XY feature set. |
| `preprocessing_1 xyz.ipynb` | Prepares the XYZ feature set. |
| `label_mapping.json` | Maps class names to encoded labels for XY models. |
| `label_mapping_xyz.json` | Maps class names to encoded labels for XYZ models. |

## Model Training

The full training notebooks compare four classifiers:

| Model | Description |
|---|---|
| KNN | Distance-based classifier that provides a simple baseline. |
| Random Forest | Tree ensemble that can model non-linear landmark patterns. |
| SVM | Effective classifier for compact normalized feature vectors. |
| XGBoost | Gradient-boosted tree model used as a powerful comparison model. |

`GridSearchCV` performs three-fold cross-validation on the training set to search over selected hyperparameters. The best estimator for each model is then evaluated on the untouched test set. The notebooks also retrain the models on augmented training data and evaluate a soft-voting ensemble that averages their predicted probabilities.

The trained models are serialized with Joblib. The original project stores them in a `models/` directory, which is kept separate from GitHub because model files can be large.

## Repository Structure

```text
.
├── app.py
├── extract_data.py
├── preprocessing_1.ipynb
├── preprocessing_1 xyz.ipynb
├── Sign_Language_Full_Implementation_XY.ipynb
├── Sign_Language_Full_Implementation_XYZ.ipynb
├── label_mapping.json
├── label_mapping_xyz.json
├── requirements.txt
├── models/                         # Download separately from Drive
├── data/                           # Download separately from Drive
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the core packages:

```bash
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python mediapipe joblib jupyter xgboost
```

If you use the original pinned `requirements.txt`, inspect it first. The file was generated on Windows and may contain platform-specific packages that are unnecessary on another operating system.

## Training the Models

1. Download the landmark CSV files from the project data storage.
2. Place the raw or prepared CSV files in the project directory.
3. Open `preprocessing_1.ipynb` for the XY pipeline or `preprocessing_1 xyz.ipynb` for the XYZ pipeline.
4. Update `CSV_PATH` and `DATA_DIR` to match your local folders.
5. Run the preprocessing cells and confirm that the label mappings and train/validation/test files are created.
6. Open the corresponding full implementation notebook.
7. Run the GridSearchCV cells to tune KNN, Random Forest, SVM, and XGBoost.
8. Review the comparison table and confusion matrices.
9. Copy or save the selected Joblib model into a local `models/` directory.

The model used by `app.py` must be trained with the same feature type, column order, normalization procedure, and label mapping used during live inference. Do not use an XYZ model with the XY feature builder or vice versa.

## Running the Webcam Application

Before launching the application, download the trained model and the matching label-mapping JSON file. Then update these variables near the top of `app.py`:

```python
MODEL_PATH = r"C:\path\to\project\models\knn_model.joblib"
LABEL_MAPPING_PATH = r"C:\path\to\project\label_mapping.json"
```

For a project-relative layout, use:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "knn_model.joblib"
LABEL_MAPPING_PATH = ROOT / "label_mapping.json"
```

Run the application with:

```bash
python app.py
```

The application opens a window named `Sign Language Recognition`. It detects one hand, draws the landmarks, shows the current predicted label, displays the word under construction, and applies a minimum confidence threshold of 50% before adding a letter.

## Keyboard Controls

Click the video window first so that it receives keyboard input.

| Key | Action |
|---|---|
| `S` | Add the current predicted letter when confidence is at least 50%. |
| `SPACE` | Add a space to the current word. |
| `BACKSPACE` | Delete the last character. |
| `C` | Clear the entire word. |
| `ENTER` | Confirm the current word in the console. |
| `Q` | Quit the application. |

Keep the keyboard layout in English while using the shortcut keys. The application includes debug output that prints the key code received by OpenCV.

## Troubleshooting

### Model file not found

Check `MODEL_PATH`, confirm that the Joblib file was downloaded, and verify that the filename matches the value used in `app.py`.

### Label mapping file not found

Download the JSON mapping that belongs to the model. An XY model must use the XY mapping, and an XYZ model must use the XYZ mapping.

### Webcam cannot be opened

Check camera permissions, close other applications that use the webcam, and try changing `cv2.VideoCapture(0)` to another camera index such as `1`.

### Predictions are incorrect

Confirm that the model and feature builder use the same coordinate format. Also verify that the hand is visible, the lighting is adequate, the correct label mapping is loaded, and the normalization code has not been changed.

### Keyboard shortcuts do nothing

Click the OpenCV video window, use an English keyboard layout, and inspect the debug key-code output printed in the terminal.

## Limitations

- The application supports one detected hand at a time.
- The prediction quality depends on the training dataset, lighting, hand orientation, camera quality, and similarity between training and live gestures.
- The current application builds a sequence of predicted letters but does not perform language-model correction or automatic word segmentation.
- Model and dataset files are stored separately and must be downloaded before running the complete system.
- The current code contains machine-specific Windows paths that should be replaced with project-relative paths.

## Future Improvements

Future work could add two-hand recognition, temporal smoothing across video frames, prediction debouncing, a confidence history, automatic word segmentation, language-model correction, multilingual output, a graphical control panel, mobile deployment, and a web-based inference API.

Evaluation should also include a subject-independent test split so that the model is measured on people it did not see during training. This provides a more realistic estimate of generalization to new users.

## Responsible Use

Sign-language recognition should be developed with representative data and with input from sign-language users and accessibility specialists. The system is an educational prototype and should not be treated as a substitute for professional interpreters or as a guaranteed communication service.

## License and Dataset Notice

Review the licenses and consent conditions for all image datasets and pretrained components before redistribution. Keep private or identifiable images out of public repositories and use the external data storage link only for datasets that may legally be shared.
