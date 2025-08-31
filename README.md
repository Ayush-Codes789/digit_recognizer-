# Handwritten Digit Recognition (MNIST + OpenCV)

This project demonstrates a **CNN model trained on the MNIST dataset** that can recognize handwritten digits.  
It supports both **retraining the model** and **using OpenCV for real-time digit recognition** (via webcam or Paint drawings).  

---

##  Project Files

- **`mnist_cnn.h5`** → Pre-trained CNN model (ready to use).  
- **`mnist.py`** → Script to train the CNN from scratch on the MNIST dataset and save the model.  
- **`opencv_model.py`** → Uses OpenCV to integrate the model with camera/screen input for real-time digit prediction.  

---

##  Installation

Clone this repository and install required dependencies:

```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install -r requirements.txt
