# ALPR-IndonesiaPlateNumber
Automatic License Plate Recognition for Indonesia Plate Number

## Method
Plate detection and character segmentation on vehicle images are using contours.<br>
For image classification, I follow [this instruction](https://www.tensorflow.org/tutorials/images/classification)

## Prerequest
- Python 3.8.6
- OpenCV 4.5.1
- NumPy 1.19.4
- Tensorflow 2.4.0

## Test Image Requirement
- Resolution 2560 x 1920
- Daylight
- The vehicle is +- 1 meter from the camera

## How to Run
- Open mainProgram.py
- Change the test image path with the same requirement image

## Retrain
- Unzip dataset.zip
- Open Training.py
- Change train dataset folder path
- You can add another character dataset to the dataset folder for better result
