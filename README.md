# 🇰🇷 Korean Handwritten Letter Classifier

A deep learning-based web application that recognizes handwritten Korean consonants (자음) in real-time. Built with PyTorch and Gradio, this project demonstrates the application of Convolutional Neural Networks (CNNs) for Korean character recognition.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Supported Characters](#supported-characters)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [Performance Metrics](#performance-metrics)
- [Educational Use](#educational-use)
- [Known Issues & Limitations](#known-issues--limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## 🎯 Overview

This project provides an interactive web interface where users can draw Korean consonants, and the model predicts which letter was drawn with confidence scores. The application uses a trained Convolutional Neural Network achieving **98% accuracy** on the test dataset.

The model can recognize 14 Korean consonants and provides real-time predictions as you draw, making it both educational and practical for Korean language learners and enthusiasts.

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/JavohirMX/korean-letter-classifier.git
cd korean-letter-classifier

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

Then open your browser at `http://127.0.0.1:7860` and start drawing!

### 📊 Project Stats

- **Model Accuracy**: 98%
- **Training Epochs**: 200
- **Data Augmentation**: 32× per image
- **Supported Characters**: 14 consonants
- **Model Size**: ~400 KB
- **Inference Speed**: <50ms

## 🛠️ Technologies Used

### Core Framework
- **PyTorch 2.9.0**: Deep learning framework for model training and inference
- **TorchVision 0.24.0**: Image transformations and augmentation
- **Gradio 5.49.1**: Interactive web interface for real-time predictions

### Image Processing
- **Pillow 11.3.0**: Image manipulation and preprocessing
- **NumPy 2.3.3**: Numerical computing and array operations

### Training & Visualization
- **Jupyter Notebook**: Interactive development and training
- **Matplotlib**: Loss and accuracy visualization
- **tqdm**: Progress bars for training monitoring

### Data Augmentation Techniques
- Random Rotation (±15°)
- Random Translation (±10%)
- Gaussian Blur (kernel=3)
- Multi-scale Cropping (4 variations)
- Combined transformations (8 per crop)

## ✨ Features

- ⚡ **Real-time Prediction**: Get instant feedback as you draw
- 🎨 **Interactive Canvas**: Clean and intuitive drawing interface (400×400 pixels)
- 🏆 **Top-5 Predictions**: View the top 5 most likely predictions with confidence scores
- 🎯 **High Accuracy**: 98% test accuracy on Korean consonant recognition
- 🌐 **Easy to Use**: Simple web interface powered by Gradio
- 💻 **CPU Compatible**: Runs on both CPU and GPU
- 📊 **Extensive Training**: 32× data augmentation with 200 training epochs
- 🔄 **Live Updates**: Predictions update automatically as you draw
- 🧹 **One-Click Clear**: Easy canvas reset for new attempts
- 📦 **Lightweight**: Only ~400 KB model size

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

## 📊 Dataset

The model is trained on a custom dataset of handwritten Korean consonants with extensive augmentation for robustness.

### Dataset Structure

The dataset contains handwritten samples for all 14 Korean consonants, organized in the following directory structure:

```
dataset/
├── ㄱ/
├── ㄴ/
├── ㄷ/
├── ㄹ/
├── ㅁ/
├── ㅂ/
├── ㅅ/
├── ㅇ/
├── ㅈ/
├── ㅊ/
├── ㅋ/
├── ㅌ/
├── ㅍ/
└── ㅎ/
```

### Data Augmentation

To improve model generalization and robustness, the following augmentation techniques are applied:

#### Geometric Transformations
- **4 Crop Variations**:
  - No crop (200×200)
  - Small crop (175×175 from 25,25)
  - Medium crop (163×163 from 50,38)
  - Large crop (163×163 from 75,38)

- **Random Rotation**: ±15 degrees
- **Random Translation**: ±10% in both x and y directions
- **Combined Rotation + Translation**: Both transformations applied

#### Image Quality Augmentations
- **Gaussian Blur**: Kernel size 3×3
- **Blur + Rotation**: Combined effect
- **Blur + Translation**: Combined effect
- **Blur + Rotation + Translation**: All three combined

### Augmentation Pipeline

Each original image undergoes:
1. **4 different crops** → 4 images
2. Each crop is augmented with **8 variations**:
   - Original (to_tensor only)
   - Rotation
   - Translation
   - Rotation + Translation
   - Blur
   - Blur + Rotation
   - Blur + Translation
   - Blur + Rotation + Translation

**Total augmentation factor**: 4 crops × 8 variations = **32× augmentation per original image**

### Dataset Split

- **Training Set**: 70% of data
- **Validation Set**: 15% of data
- **Test Set**: 15% of data
- **Random Seed**: 42 (for reproducibility)

### Preprocessing

All images are:
- Converted to grayscale (L mode)
- Resized to **100×100 pixels**
- Normalized to **[0, 1]** range using PyTorch's `ToTensor()`

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

### Layer Details

| Layer | Type | Input → Output | Kernel | Parameters |
|-------|------|----------------|--------|------------|
| 1 | Conv2d | 1→32 | 3×3 | 320 |
| 2 | MaxPool2d | - | 2×2 | 0 |
| 3 | Conv2d | 32→64 | 3×3 | 18,496 |
| 4 | MaxPool2d | - | 2×2 | 0 |
| 5 | Conv2d | 64→128 | 3×3 | 73,856 |
| 6 | AdaptiveAvgPool2d | - | 1×1 | 0 |
| 7 | Dropout | - | p=0.25 | 0 |
| 8 | Linear | 128→64 | - | 8,256 |
| 9 | Dropout | - | p=0.3 | 0 |
| 10 | Linear | 64→14 | - | 910 |

**Total Trainable Parameters**: ~101,838

## 🎯 Training Process

The model training follows a rigorous process documented in `korean_letter_classifier.ipynb`.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 200 |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 (1e-3) |
| **Loss Function** | CrossEntropyLoss |
| **Device** | CUDA (if available) / CPU |

### Training Procedure

1. **Data Loading**: 
   - Unzip `dataset.zip`
   - Load and augment images
   - Create train/val/test splits (70/15/15)
   - Initialize DataLoaders with batch_size=32

2. **Model Initialization**:
   - Define CNN architecture
   - Move model to GPU (if available)
   - Initialize Adam optimizer with lr=1e-3

3. **Training Loop** (200 epochs):
   ```python
   For each epoch:
     - Train phase:
       * Forward pass through training data
       * Calculate loss (CrossEntropyLoss)
       * Backward propagation
       * Update weights with Adam optimizer
       * Track training loss and accuracy
     
     - Validation phase:
       * Evaluate on validation set (no gradient)
       * Calculate validation loss and accuracy
       * Track metrics for analysis
   ```

4. **Monitoring**:
   - Progress bar with tqdm
   - Real-time loss display
   - Track train/val losses and accuracies

5. **Evaluation**:
   - Test on held-out test set
   - Calculate final accuracy
   - Generate loss and accuracy curves

### Training Results

- **Final Test Accuracy**: 98%
- **Model saved as**: `modelT98.pth`
- **Training visualization**: Loss and accuracy curves available in notebook

### How to Train Your Own Model

1. **Prepare the dataset**:
   ```bash
   # Ensure dataset.zip is in the project root
   unzip dataset.zip
   ```

2. **Open the training notebook**:
   ```bash
   jupyter notebook korean_letter_classifier.ipynb
   ```
   Or use VS Code with Jupyter extension

3. **Run all cells sequentially**:
   - Dataset preparation and augmentation
   - Model definition
   - Training loop
   - Evaluation and visualization
   - Model saving

4. **The trained model will be saved as** `modelT{accuracy}.pth`

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

### Training Your Own Model

If you want to train the model from scratch or with your own dataset:

1. **Extract the dataset**:
   ```bash
   unzip dataset.zip
   ```

2. **Open and run the training notebook**:
   ```bash
   jupyter notebook korean_letter_classifier.ipynb
   ```
   Or use VS Code with the Jupyter extension

3. **Follow the notebook sections**:
   - Dataset Preparation and Augmentation
   - Model Definition and Training
   - Results and Evaluation
   - Model Saving

The trained model will be automatically saved as `modelT{accuracy}.pth`

## 📁 Project Structure

```
korean-letter-classifier/
│
├── main.py                          # Main application file with Gradio interface
├── korean_letter_classifier.ipynb   # Training notebook with full pipeline
├── modelT98.pth                     # Trained model weights (98% accuracy)
├── dataset.zip                      # Compressed dataset of Korean letters
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation (this file)
```

### File Descriptions

- **`main.py`**: Contains the CNN model architecture, preprocessing functions, prediction logic, and Gradio interface for real-time inference
- **`korean_letter_classifier.ipynb`**: Complete training pipeline including data augmentation, model training, evaluation, and visualization
- **`modelT98.pth`**: Pre-trained PyTorch model weights achieving 98% test accuracy
- **`dataset.zip`**: Compressed archive containing handwritten Korean consonant images organized by character
- **`requirements.txt`**: All Python package dependencies with specific versions

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

### Training Summary

The model was trained with extensive data augmentation:
- **Input Size**: 100×100 grayscale images
- **Batch Size**: 32
- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: CrossEntropyLoss
- **Training Epochs**: 200
- **Data Augmentation**: 32× per image (4 crops × 8 variations)
  - Rotation (±15°)
  - Translation (±10%)
  - Gaussian Blur (kernel=3)
  - Combined transformations
- **Dataset Split**: 70% train / 15% validation / 15% test
- **Final Test Accuracy**: 98%
- **Regularization**: Dropout (0.25, 0.3) to prevent overfitting

## 📦 Requirements

### Core Dependencies

- **torch** (2.9.0+cpu): Deep learning framework
- **torchvision** (0.24.0+cpu): Image processing utilities
- **gradio** (5.49.1): Web interface framework
- **Pillow** (11.3.0): Image processing
- **numpy** (2.3.3): Numerical computing

### Full Dependencies

See [`requirements.txt`](requirements.txt) for the complete list of dependencies.

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.00% |
| **Model Size** | ~400 KB |
| **Parameters** | ~101K |
| **Inference Time** | <50ms (CPU) |
| **Training Time** | ~1-2 hours (GPU) |
| **Input Resolution** | 100×100 pixels |
| **Output Classes** | 14 consonants |

### Model Performance

- **Precision**: High precision across all 14 classes
- **Recall**: Consistent recall performance
- **Generalization**: Robust to different handwriting styles due to extensive augmentation
- **Real-time**: Fast enough for interactive use

## 🎓 Educational Use

This project is ideal for:

- **Korean Language Learners**: Practice writing Korean consonants with instant feedback
- **Machine Learning Students**: Study CNN architectures, data augmentation, and image classification
- **Computer Vision Projects**: Complete example of handwritten character recognition pipeline
- **Deep Learning Demonstrations**: Interactive ML application with training notebook showcase
- **Research & Development**: Foundation for Korean text recognition systems

## 🐛 Known Issues & Limitations

- Model is trained on specific handwriting styles; may vary with different writing styles
- Best performance with clear, centered drawings
- Only recognizes consonants (자음), not vowels (모음) or complete syllable blocks
- Requires consistent stroke width and style

## 🔧 Troubleshooting

### Common Issues

**Issue**: Model predictions are inaccurate
- **Solution**: Draw clearly in the center of the canvas
- **Solution**: Use consistent stroke width
- **Solution**: Ensure the character is large enough

**Issue**: Application won't start
- **Solution**: Verify all dependencies are installed: `pip install -r requirements.txt`
- **Solution**: Check that `modelT98.pth` exists in the project directory
- **Solution**: Ensure Python 3.8+ is being used

**Issue**: CUDA/GPU errors
- **Solution**: The model automatically falls back to CPU if GPU is unavailable
- **Solution**: Update PyTorch: `pip install --upgrade torch torchvision`

**Issue**: Port already in use
- **Solution**: Modify the port in `main.py`: `demo.launch(server_port=8080)`

### Training Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in the notebook (e.g., from 32 to 16)
- **Solution**: Use CPU instead of GPU for training

**Issue**: Dataset not found
- **Solution**: Ensure `dataset.zip` is extracted: `unzip dataset.zip`
- **Solution**: Verify the `dataset/` folder exists with 14 subfolders

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
- [ ] Transfer learning with pre-trained models
- [ ] Ensemble methods for improved accuracy
- [ ] Real-time writing feedback and stroke analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- **Dataset Expansion**: Add more handwriting samples
- **Model Improvements**: Experiment with different architectures
- **Feature Addition**: Implement vowel or syllable recognition
- **Documentation**: Improve code comments and documentation
- **UI/UX**: Enhance the Gradio interface
- **Performance**: Optimize inference speed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Authors

- [JavohirMX](https://github.com/JavohirMX)
- Son Lee  


## 🙏 Acknowledgments

- Yonsei University - Artificial Intelligence Course (2025-2)
- Korean language and Hangul writing system
- PyTorch and Gradio communities
- Open-source contributors

## 📧 Contact

For questions, suggestions, or issues, please:
- Open an issue on GitHub
- Contact via GitHub profile

---

**Made with ❤️ for Korean language learners and ML enthusiasts**

*Last Updated: November 2025*
