import gradio as gr
import torch
from torch import nn
from PIL import Image
import numpy as np

# Define the model architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 14)
).to(device)

# Load the trained weights
model.load_state_dict(torch.load('modelT98.pth', map_location=device))
model.eval()

# Class labels mapping
class_labels = {
    0: 'ㄹ', 1: 'ㅋ', 2: 'ㄱ', 3: 'ㅈ', 4: 'ㅅ', 5: 'ㅌ', 6: 'ㅁ',
    7: 'ㅊ', 8: 'ㅇ', 9: 'ㅍ', 10: 'ㄴ', 11: 'ㅎ', 12: 'ㅂ', 13: 'ㄷ'
}

def preprocess_image(image):
    """Preprocess the drawn image for model input"""
    if image is None:
        return None
    
    # Handle dict format from Sketchpad (contains 'background' and 'layers')
    if isinstance(image, dict):
        # Get the composite image from the sketchpad
        if 'composite' in image:
            image = image['composite']
        elif 'background' in image:
            image = image['background']
        else:
            return None
    
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # At this point, ensure we have a PIL Image
    if not hasattr(image, 'mode'):
        return None
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 100x100
    image = image.resize((100, 100), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1] to match ToTensor() from training
    img_array = np.array(image).astype('float32') / 255.0
    
    # If the canvas uses white background with black strokes, MNIST expects white (foreground) on black background.
    # Many Sketchpads produce black strokes on white background; if digits look inverted, uncomment the next line
    # img_array = 1.0 - img_array

    # Convert to tensor and add channel and batch dimensions -> shape (1, 1, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float().to(device)

    return img_tensor

def predict(image):
    """Make prediction on the drawn image"""
    if image is None:
        return {}
    
    # Preprocess the image
    img_tensor = preprocess_image(image)
    
    if img_tensor is None:
        return {}

    # Ensure tensor is 4D: (N, C, H, W). If not, try to reshape safely.
    if img_tensor.dim() == 3:
        # add batch dim
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 4:
        return {}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Verify probabilities sum to ~1.0 (for debugging)
    # prob_sum = probabilities.sum().item()
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, min(5, len(class_labels)))
    
    # Format results - normalize to percentages properly
    results = {}
    for prob, idx in zip(top5_prob, top5_idx):
        letter = class_labels[idx.item()]
        confidence = float(prob.item())  # Keep as 0-1 range, Gradio will convert to %
        results[f"{letter}"] = confidence
    
    return results

# Create Gradio interface
with gr.Blocks(title="Korean Handwritten Letter Classifier") as demo:
    gr.Markdown(
        """
        # 🇰🇷 Korean Handwritten Letter Classifier
        Draw a Korean consonant (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ) and the model will predict it!
        """
    )
    
    with gr.Row():
        with gr.Column():
            # Drawing canvas
            canvas = gr.Sketchpad(
                label="Draw a Korean Letter Here",
                type="pil",
                image_mode="L",
                canvas_size=(400, 400),
                brush=gr.Brush(default_size=8, colors=["#000000"], color_mode="fixed")
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear Canvas", variant="secondary")
                predict_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            # Prediction output
            output = gr.Label(
                label="Predictions (Top 5)",
                num_top_classes=5
            )
            
            gr.Markdown(
                """
                ### Supported Letters:
                ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
                """
            )
    
    # Event handlers
    predict_btn.click(fn=predict, inputs=canvas, outputs=output)
    canvas.change(fn=predict, inputs=canvas, outputs=output)  # Real-time prediction
    # Clear the sketch by returning an empty PIL image matching the canvas size
    def clear_canvas():
        # Return a blank white image in 'L' mode so the Sketchpad is simply cleared
        return Image.new('L', (400, 400), color=255)

    clear_btn.click(fn=clear_canvas, outputs=canvas)
    
    gr.Markdown(
        """
        ---
        **Tips:** 
        - Draw clearly in the center of the canvas
        - The model predicts in real-time as you draw
        - Use the Clear button to start over
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)