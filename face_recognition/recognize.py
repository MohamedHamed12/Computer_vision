import json
import sys
import cv2 as cv
import os
import numpy as np
import pathlib
from typing import List, Tuple, Dict



def recognize_faces(test_image_path: str) -> np.ndarray:
    """
    Recognizes faces in the given test image and returns the image with labeled faces.

    Args:
        test_image_path (str): The path to the test image.

    Returns:
        np.ndarray: The test image with labeled faces.
    """
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the test image
    test_image = cv.imread(test_image_path)

    # Convert the test image to grayscale
    gray_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_roi = gray_image[y:y+h, x:x+w]

        # Use the face recognizer to predict the label for the face region
        label, confidence = face_recognizer.predict(face_roi)

        # Get the corresponding label from the label mapping (if used during training)
        label_str = [key for key, value in label_mapping.items() if value == label][0]

        # Draw the rectangle and label on the detected face
        cv.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(test_image, f'{label_str} ({confidence:.2f})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return test_image


def get_label_mapping(label_mapping_file: str) -> Dict[str, int]:
    """
    Get the label mapping from a JSON file.

    Args:
        label_mapping_file (str): The path to the JSON file containing the label mapping.

    Returns:
        Dict[str, int]: A dictionary mapping string labels to integer IDs.
    """
    with open(label_mapping_file, 'r') as f:
        label_mapping = json.load(f)

    return label_mapping

def main():
    global face_recognizer, label_mapping
    cur_path=str(pathlib.Path(__file__).parent.absolute())

    label_mapping=get_label_mapping(str(cur_path) + '/data/label_mapping.json')
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    trained_model_path = os.path.join(str(cur_path), 'data', 'trained_model.xml')

    if os.path.exists(trained_model_path):
        face_recognizer.read(trained_model_path)

    else:
        print("Trained model not found. Please train the model first.")
        exit(1)

    # Set the path to the test image
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = str(cur_path) + '/data/test_image.png'

    # Recognize faces in the test image
    result_image = recognize_faces(test_image_path)

    # Display the test image with the detected faces and labels
    cv.imshow("Result Image", result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    sys.exit(main())