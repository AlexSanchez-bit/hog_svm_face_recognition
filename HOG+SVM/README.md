# Face Recognition System

This project implements a live face recognition system using a combination of Histogram of Oriented Gradients (HOG) for face detection and a Support Vector Machine (SVM) for facial recognition. The system is capable of detecting faces in a live video stream and identifying individuals based on a pre-trained SVM model.

## Features

-   **Live Face Detection**: Utilizes HOG descriptors to detect faces in real-time from a video feed.
-   **Face Recognition**: Employs a trained SVM model to recognize individuals from detected faces.
-   **Training Script**: Includes a script (`svm_training.py`) to train the SVM model using a dataset (e.g., Labeled Faces in the Wild - LFW).
-   **Video Capture Integration**: Integrates with video capture devices to process live streams.

## Requirements

The project relies on several Python libraries. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/face_recognition.git
    cd face_recognition
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset**: Ensure you have the LFW dataset (or a similar face dataset) available. The current structure suggests it's located in `lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/`. If not present, you will need to download and structure it accordingly for the `svm_training.py` script.

    The expected structure for the LFW dataset is:
    ```
    lfw-dataset/
    └── lfw-deepfunneled/
        └── lfw-deepfunneled/
            ├── Person_Name_01/
            │   ├── Person_Name_01_0001.jpg
            │   └── ...
            ├── Person_Name_02/
            │   ├── Person_Name_02_0001.jpg
            │   └── ...
            └── ...
    ```

## Usage

### 1. Train the SVM Model

To train the face recognition model, run the `svm_training.py` script. This script will process the images from your dataset, extract HOG features, and train an SVM classifier.

```bash
python svm_training.py
```
*(Note: You might need to adjust paths within `svm_training.py` if your dataset is located elsewhere or if the script doesn't automatically find it.)*

### 2. Live Face Recognition

After training, you can use the `video_capture.py` script to perform live face detection and recognition. This script will open your webcam feed, detect faces, and attempt to identify them using the trained SVM model.

```bash
python video_capture.py
```

### 3. Face Detection (Standalone)

The `facedetection.py` script might contain functions solely for face detection using HOG, without the recognition component. You can investigate its contents for standalone detection functionalities.

```bash
python facedetection.py # (If it's designed to be run directly)
```

## Project Structure

-   `HOG.py`: Contains functions related to Histogram of Oriented Gradients (HOG) feature extraction.
-   `facedetection.py`: Implements face detection logic, likely utilizing the HOG features.
-   `svm_training.py`: Script for training the Support Vector Machine (SVM) model for face recognition.
-   `video_capture.py`: Handles live video stream capture, integrates face detection, and recognition.
-   `requirements.txt`: Lists all Python dependencies.
-   `lfw-dataset/`: Directory containing the Labeled Faces in the Wild (LFW) dataset, used for training.
