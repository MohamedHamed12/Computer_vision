import json
import sys
import cv2 as cv
import os
import numpy as np
import pathlib
from typing import List, Tuple, Dict


def get_images_and_labels(data_folder: str) -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    """
    Get images, labels, and label mapping from a given data folder.
    
    Args:
    - data_folder: The path to the folder containing the images
    
    Returns:
    - images: A list of numpy arrays representing the images
    - labels: A list of integers representing the labels
    - label_mapping: A dictionary mapping string labels to integer IDs
    """
    image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    images = []
    labels = []
    label_mapping = {}
    current_label_id = 0

    for image_path in image_paths:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        label_str = os.path.split(image_path)[-1].split(".")[0]

        if label_str not in label_mapping:
            label_mapping[label_str] = current_label_id
            current_label_id += 1

        label = label_mapping[label_str]
        images.append(img)
        labels.append(label)

    return images, labels, label_mapping


def save_label_mapping(label_mapping: Dict[str, int], output_file: str) -> None:
    """
    Save the label mapping to a JSON file.

    Args:
    - label_mapping: A dictionary mapping string labels to integer IDs
    - output_file: The path to the output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(label_mapping, f)

    print(f"Label mapping saved to {output_file}.")


def main():
    global face_recognizer, label_mapping
    cur_path=str(pathlib.Path(__file__).parent.absolute()) 
    data_path=cur_path + '/data/single_images'

    images, labels, label_mapping = get_images_and_labels(data_path)

    save_label_mapping(label_mapping, cur_path + '/data/label_mapping.json')
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(images, np.array(labels))
    #create if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    face_recognizer.save('data/trained_model.xml')

    print("Training complete.")


if __name__ == '__main__':
    sys.exit(main())