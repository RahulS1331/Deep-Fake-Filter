# Deep Fake Filter

## Project Description

The Deep Fake Filter project demonstrates a computer vision application that overlays a pre-defined celebrity face onto another face detected in a live video feed. Using Python libraries like OpenCV and dlib, the project involves real-time face detection, facial landmark identification, image transformation, and seamless blending to create a convincing deep fake effect.

## Objectives

- To learn and apply computer vision techniques for real-time applications.
- To create an engaging application for fun and educational purposes.
- To showcase practical skills in Python programming, OpenCV, and dlib.

## Key Features

- **Real-time Face Detection:** Using dlib’s pre-trained face detector.
- **Facial Landmark Detection:** Identifying key points on the detected face.
- **Image Transformation:** Aligning the celebrity face with the detected face.
- **Seamless Blending:** Using OpenCV’s seamlessClone for natural image blending.
- **Live Video Processing:** Applying the filter in real-time on a video feed.

## Installation

To get started with the Deep Fake Filter project, follow these steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/DeepFake-Filter.git
    cd DeepFake-Filter
    ```

2. Install the required libraries:

    ```sh
    pip install opencv-python dlib numpy
    ```

3. Download the shape predictor model and place it in the `data` directory:

    - [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

4. Place a celebrity image in the `images` directory.

## Usage

1. Open the Jupyter Notebook in the `notebooks` directory:

    ```sh
    jupyter notebook notebooks/DeepFake_Filter.ipynb
    ```

2. Run the notebook to see the deep fake filter in action.

Alternatively, you can run the provided Python script for real-time video processing:

```python
import cv2
import dlib
import numpy as np

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
celebrity_img = cv2.imread("images/celebrity.jpg")

def apply_filter(frame, celebrity_img, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], np.int32)
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        celebrity_resized = cv2.resize(celebrity_img, (w, h))
        mask = 255 * np.ones(celebrity_resized.shape, celebrity_resized.dtype)
        center = (x + w // 2, y + h // 2)
        frame = cv2.seamlessClone(celebrity_resized, frame, mask, center, cv2.NORMAL_CLONE)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = apply_filter(frame, celebrity_img, detector, predictor)
    cv2.imshow("Deep Fake Filter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
