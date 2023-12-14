import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib

cur_path=str(pathlib.Path(__file__).parent.absolute())
default_input_path=cur_path + '/data/input_images'
default_output_path=cur_path + '/data/output_images'


class DetectImage:
    def __init__(self, image_path, output_path=default_output_path):
        self.image_path = image_path
        self.output_path = output_path     
      
    def detect_image(self):
        self.img = cv.imread(self.image_path)
        self.img= self.resize_image()
        thresh = self.threshold_image()
        self.get_lines(thresh)
        self.get_words(thresh)

    
    def resize_image(self,new_width = 1000):

        height, width = self.img.shape[:2]
        aspect_ratio = width / height
        new_height = int(new_width / aspect_ratio)
        resized_img = cv.resize(self.img, (new_width, new_height))
        return resized_img

    def threshold_image(self):
        # Convert the image to grayscale
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # Apply thresholding to the grayscale image
        _, thresh = cv.threshold(gray,  80,255, cv.THRESH_BINARY_INV )

        return thresh

    def show_contours(self, contours, name):
        img=self.img.copy()
        for ctr in contours:
            
            x,y,w,h = cv.boundingRect(ctr)
            cv.rectangle(img, (x,y), (x+w, y+h), (40, 100, 250), 2)
            
        cv.imshow('img', img); cv.waitKey(0)
        #save image
        cv.imwrite( self.output_path + '/'  + name+'_' + os.path.basename(self.image_path) , img)

    def get_lines(self,thresh):

        kernel = np.ones((3,85), np.uint8)
        # kernel = np.ones((3, 30), np.uint8)
        dilated = cv.dilate(thresh, kernel, iterations = 1)
        morph_img = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

        (contours, heirarchy) = cv.findContours(morph_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(contours, key = lambda ctr : cv.boundingRect(ctr)[1]) 
        self.show_contours( sorted_contours_lines, 'lines')



    def get_words(self,thresh): 

        kernel = np.ones((3,15), np.uint8)
    
        morph_img = cv.dilate(thresh, kernel, iterations = 1)
        # morph_img = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

        (contours, heirarchy) = cv.findContours(morph_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        sorted_contours_words = sorted(contours, key = lambda ctr : cv.boundingRect(ctr)[0]) 

        self.show_contours( sorted_contours_words, 'words')                  

    
  

def main():
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_input_path

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:

        #make output folder
        output_path = default_output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    # get list of images
    image_paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if   file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:
        img_detect=DetectImage(image_path, output_path)
        img_detect.detect_image()

    return 0

if __name__ == '__main__':
    # Load the image
    sys.exit(main())
    