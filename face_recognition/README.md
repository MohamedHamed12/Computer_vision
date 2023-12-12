# Face Recognition using OpenCV

This project demonstrates basic face recognition using OpenCV, a popular computer vision library in Python.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Credits](#credits)
- [License](#license)

## Overview

This project uses OpenCV to perform face recognition. It trains a Local Binary Pattern Histogram (LBPH) face recognizer on a dataset of labeled face images and then uses the recognizer to predict the labels of faces in a test image.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MohamedHamed12/Computer_vision
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Organize your data:

   Place your training face images in the `data/single_images` folder. Each image file should be named with the format `label.jpg` (e.g., `karim_benzem.jpg`)

2. Train the face recognizer:

   ```bash
   python train.py
   ```

   This will train the LBPH face recognizer on your dataset.

3. Test the recognizer:

   Replace `path/to/test_image.jpg` with the path to the test image you want to use.

   ```bash
   python recognize.py path/to/test_image.jpg
   ```

   The recognized faces will be displayed in the output image.

## File Structure

```
face-recognition-opencv/
│
├── data
│   ├──single_images/                 # Training images folder
│   ├──├── karim_benzem.jpg
│   ├──├── ....
│
├── ├── trained_model.yml     # Trained face recognizer model
├── ├──haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── train.py              # Script to train the face recognizer
├── recognize.py          # Script to recognize faces in a test image
├── README.md             # Project documentation
```

## Example
![Alt text](image.png)
## Credits

- This project was created by [Mohamed Hamed](https://github.com/MohamedHamed12)

## License

This project is licensed under the [MIT License](LICENSE).

