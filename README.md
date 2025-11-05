# 🇰🇷 Korean Handwritten Letter Classifier

A deep learning-based web application that recognizes handwritten Korean consonants (자음) in real-time. Built with PyTorch and Gradio, this project demonstrates the application of Convolutional Neural Networks (CNNs) for Korean character recognition.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Characters](#supported-characters)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project provides an interactive web interface where users can draw Korean consonants, and the model predicts which letter was drawn with confidence scores. The application uses a trained Convolutional Neural Network achieving 98% accuracy on the test dataset.

The model can recognize 14 Korean consonants and provides real-time predictions as you draw, making it both educational and practical for Korean language learners and enthusiasts.

## ✨ Features

- **Real-time Prediction**: Get instant feedback as you draw
- **Interactive Canvas**: Clean and intuitive drawing interface
- **Top-5 Predictions**: View the top 5 most likely predictions with confidence scores
- **High Accuracy**: 98% test accuracy on Korean consonant recognition
- **Easy to Use**: Simple web interface powered by Gradio
- **CPU Compatible**: Runs on both CPU and GPU

## 🔤 Supported Characters

The model can recognize the following 14 Korean consonants (자음):

| Character | Name | Romanization |
|-----------|------|--------------|
| ㄱ | 기역 | giyeok (g/k) |
| ㄴ | 니은 | nieun (n) |
| ㄷ | 디귿 | digeut (d/t) |
| ㄹ | 리을 | rieul (r/l) |
| ㅁ | 미음 | mieum (m) |
| ㅂ | 비읍 | bieup (b/p) |
| ㅅ | 시옷 | siot (s) |
| ㅇ | 이응 | ieung (ng/silent) |
| ㅈ | 지읒 | jieut (j) |
| ㅊ | 치읓 | chieut (ch) |
| ㅋ | 키읔 | kieuk (k) |
| ㅌ | 티읕 | tieut (t) |
| ㅍ | 피읖 | pieup (p) |
| ㅎ | 히읗 | hieut (h) |

## 🏗️ Model Architecture

The classifier uses a custom Convolutional Neural Network with the following architecture:

```
Input (1x100x100) 
    ↓
Conv2d(1→32, 3x3) + ReLU + MaxPool(2x2)
    ↓
Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2)
    ↓
Conv2d(64→128, 3x3) + ReLU
    ↓
AdaptiveAvgPool2d(1x1)
    ↓
Dropout(0.25) + Flatten
    ↓
Linear(128→64) + ReLU + Dropout(0.3)
    ↓
Linear(64→14)
    ↓
Output (14 classes)
```

**Key Features:**
- 3 Convolutional layers with increasing depth (32 → 64 → 128)
- Max pooling for spatial dimension reduction
- Adaptive average pooling for global features
- Dropout layers (0.25, 0.3) for regularization
- **Total Parameters**: ~18K
- **Accuracy**: 98% on test set

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JavohirMX/korean-letter-classifier.git
   cd korean-letter-classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   
   Ensure `modelT98.pth` is present in the project root directory. This file contains the trained model weights.

## 💻 Usage

### Running the Application

1. **Start the Gradio interface**
   ```bash
   python main.py
   ```

2. **Access the web interface**
   
   The application will start a local server. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

3. **Using the classifier**
   - Draw a Korean consonant on the canvas using your mouse or touchscreen
   - The model will automatically predict as you draw
   - View the top 5 predictions with confidence scores
   - Click "Clear Canvas" to start over

### Command Line Options

To share the app publicly (generates a temporary public URL):
```python
# Modify main.py, line 177
demo.launch(share=True)
```

To run on a specific port:
```python
demo.launch(server_port=8080)
```

## 📁 Project Structure

```
korean-letter-classifier/
│
├── main.py              # Main application file with model and Gradio interface
├── modelT98.pth         # Trained model weights (98% accuracy)
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation (this file)
```

### File Descriptions

- **`main.py`**: Contains the CNN model architecture, preprocessing functions, prediction logic, and Gradio interface
- **`modelT98.pth`**: Pre-trained PyTorch model weights (98% test accuracy)
- **`requirements.txt`**: All Python package dependencies with versions

## 🔧 Technical Details

### Image Preprocessing

The application performs the following preprocessing steps:

1. **Input Handling**: Accepts PIL Image or numpy array from Gradio Sketchpad
2. **Grayscale Conversion**: Converts to single-channel grayscale (L mode)
3. **Resizing**: Resizes to 100×100 pixels using LANCZOS interpolation
4. **Normalization**: Scales pixel values to [0, 1] range
5. **Tensor Conversion**: Converts to PyTorch tensor with shape (1, 1, 100, 100)

### Prediction Pipeline

```python
User Drawing → PIL Image → Grayscale → Resize(100×100) → 
Normalize → Tensor → CNN → Softmax → Top-5 Predictions
```

### Model Training

The model was trained with:
- **Input Size**: 100×100 grayscale images
- **Batch Size**: Optimized for training performance
- **Optimizer**: Adam or SGD (with momentum)
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: Rotation, translation, scaling (likely)
- **Final Test Accuracy**: 98%

## 📦 Requirements

### Core Dependencies

- **torch** (2.9.0+cpu): Deep learning framework
- **torchvision** (0.24.0+cpu): Image processing utilities
- **gradio** (5.49.1): Web interface framework
- **Pillow** (11.3.0): Image processing
- **numpy** (2.3.3): Numerical computing

### Full Dependencies

See [`requirements.txt`](requirements.txt) for the complete list of dependencies.

## 🎓 Educational Use

This project is ideal for:

- **Korean Language Learners**: Practice writing Korean consonants
- **Machine Learning Students**: Study CNN architectures and image classification
- **Computer Vision Projects**: Example of handwritten character recognition
- **Deep Learning Demonstrations**: Interactive ML application showcase

## 🐛 Known Issues & Limitations

- Model is trained on specific handwriting styles; may vary with different writing styles
- Best performance with clear, centered drawings
- Only recognizes consonants (자음), not vowels (모음) or complete syllable blocks
- Requires consistent stroke width and style

## 🔮 Future Improvements

Potential enhancements for future versions:

- [ ] Add support for Korean vowels (모음)
- [ ] Recognize complete Korean syllable blocks (한글)
- [ ] Improve accuracy with more training data
- [ ] Add data augmentation during inference
- [ ] Support for mobile-optimized interface
- [ ] Export predictions as text
- [ ] Multi-character sequence recognition
- [ ] Model explanation/visualization features


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**JavohirMX**
- GitHub: [@JavohirMX](https://github.com/JavohirMX)


## 📧 Contact

For questions, suggestions, or issues, please:
- Open an issue on GitHub
- Contact via GitHub profile

---

**Made with ❤️ for Korean language learners and ML enthusiasts**

*Last Updated: November 2025*
