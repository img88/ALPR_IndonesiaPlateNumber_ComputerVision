# ALPR-IndonesiaPlateNumber
Automatic License Plate Recognition for Indonesia Plate Number<br>

## Method
Plate detection and character segmentation on vehicle images are using contours.<br>
For image/character classification, I follow [this instruction](https://www.tensorflow.org/tutorials/images/classification)

### Good Result

<p align="center">
  <img src="https://user-images.githubusercontent.com/56859155/105789180-89779680-5fb4-11eb-9671-b2ae356a4cb2.png" /><br>
  <img src="https://user-images.githubusercontent.com/56859155/105789470-13bffa80-5fb5-11eb-8c43-b19b238ca961.jpg" />
  <img src="https://user-images.githubusercontent.com/56859155/105787942-0d7c4f00-5fb2-11eb-96ee-3f42c8b242c4.png" />
</p>

### Ok Result

<p align="center">
  <img src="https://user-images.githubusercontent.com/56859155/105789213-9a280c80-5fb4-11eb-9ea7-4af89f555631.png"/><br>
  <img src="https://user-images.githubusercontent.com/56859155/105787793-c42bff80-5fb1-11eb-8eb8-468d4ccebff4.jpg"/>
  <img src="https://user-images.githubusercontent.com/56859155/105789172-84b2e280-5fb4-11eb-90d8-2c2783c6c6bd.png"/>
</p>

### Bad Result

<p align="center">
  <img src="https://user-images.githubusercontent.com/56859155/105787785-c1c9a580-5fb1-11eb-8d71-4555b1aadf15.jpg"/>
  <img src="https://user-images.githubusercontent.com/56859155/105787797-c5f5c300-5fb1-11eb-8e6b-12ef7b108ab1.jpg"/>
  <img src="https://user-images.githubusercontent.com/56859155/105787801-c68e5980-5fb1-11eb-89e4-84bf50d802ae.jpg"/>
</p>

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
- Change the test image path with the same image requirement

## Retrain
- Unzip dataset.zip
- Open Training.py
- Change train dataset folder path
- You can add another character dataset to the dataset folder for better result
