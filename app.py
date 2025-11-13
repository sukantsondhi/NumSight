"""
Flask API Server for MNIST Digit Recognition

This server provides endpoints for:
- Uploading and predicting handwritten digits
- Serving the frontend application
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import base64

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the trained model
MODEL_PATH = 'models/mnist_model.h5'
model = None

def load_model():
    """Load the trained MNIST model."""
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")

def preprocess_image(image):
    """
    Preprocess an image for model prediction.
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array: Preprocessed image ready for prediction
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Invert if needed (MNIST digits are white on black background)
    # Check if the image has more white pixels than black
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1) for model input
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')

@app.route('/style.css')
def style():
    """Serve the CSS file."""
    return send_from_directory('static', 'style.css')

@app.route('/script.js')
def script():
    """Serve the JavaScript file."""
    return send_from_directory('static', 'script.js')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the digit from an uploaded image.
    
    Expected request format:
    - JSON with 'image' field containing base64 encoded image data
    
    Returns:
        JSON response with prediction and confidence scores
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode and open image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit])
        
        # Get all probabilities
        probabilities = {
            str(i): float(predictions[0][i]) 
            for i in range(10)
        }
        
        return jsonify({
            'digit': int(predicted_digit),
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load the model
    load_model()
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the server
    print("=" * 60)
    print("MNIST Digit Recognition Server")
    print("=" * 60)
    print("Server running on http://localhost:5000")
    print("Open http://localhost:5000 in your browser to use the app")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
