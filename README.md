# Morphos

![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-3.18.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MobileNet](https://img.shields.io/badge/MobileNet-v2-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.0-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)
[![Deployed on Netlify](https://img.shields.io/badge/Deployed%20on-Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://morphos.netlify.app/)
![morphos netlify app_](https://github.com/user-attachments/assets/43a36f1c-6a49-44bf-939b-62bcbbb3ed4d)


## Overview

Morphos is a web-based platform that allows users to **train custom image classification models** without writing code. Built with TensorFlow.js and MobileNet, it provides an intuitive interface for creating, training, and deploying machine learning models directly in the browser. Users can add image samples via webcam or file upload, train models with customizable hyperparameters, and use them for real-time predictions.


### Features

- **No-Code Model Training**: Create and train image classification models using a simple UI.
- **Webcam & Upload Support**: Add training samples via webcam or image uploads.
- **Real-Time Predictions**: Preview model predictions in real-time using a webcam feed.
- **Model Export**: Export trained models as ZIP files for later use.
- **Customizable Hyperparameters**: Adjust epochs, batch size, and learning rate for training.
- **Browser-Based**: Runs entirely in the browser with TensorFlow.js, no server-side processing required.

### Model Details

- **Base Model**: MobileNet v2 (alpha: 0.5) for feature extraction
- **Custom Layers**: 
  - Dense (128 units, ReLU) + Dropout (0.2)
  - Dense (numClasses, Softmax)
- **Default Hyperparameters**:
  - **Epochs**: 10 (configurable: 1-100)
  - **Batch Size**: 16 (configurable: 1-64)
  - **Learning Rate**: 0.001 (configurable: 0.0001-0.1)
- **Input Size**: 224x224 pixels
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

### How It Works

Morphos uses MobileNet v2 as a pre-trained feature extractor, adding custom dense layers for classification. Users define classes, add image samples, and train the model in-browser. The trained model can be exported as a ZIP file (containing `model.json`, `weights.bin`, and `class_names.csv`) and loaded later for predictions via webcam or single images.

### Deployment

You can interact with the app via the following link:

https://morphos.netlify.app

## License

This project is licensed under the MIT License.

