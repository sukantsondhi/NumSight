"""
Flask API Server for MNIST Digit Recognition

This server provides endpoints for:
- Uploading and predicting handwritten digits
- Serving the frontend application
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
import io
import os
import base64

app = Flask(__name__, static_folder='docs')
CORS(app)

# Load the trained model
MODEL_DIR = 'models'
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, 'mnist_model.keras')
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'mnist_model.h5')
model = None

def load_model():
    """Load the trained MNIST model."""
    global model
    # Prefer modern Keras format if available, otherwise fall back to legacy HDF5
    if os.path.exists(MODEL_PATH_KERAS):
        model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        print(f"Model loaded from {MODEL_PATH_KERAS}")
    elif os.path.exists(MODEL_PATH_H5):
        model = tf.keras.models.load_model(MODEL_PATH_H5)
        print(f"Model loaded from {MODEL_PATH_H5}")
        print("Note: Consider re-saving the model in .keras format for future compatibility.")
    else:
        print("Warning: No model file found in 'models/'.")
        print("Expected one of: 'models/mnist_model.keras' or 'models/mnist_model.h5'")
        print("Please train the model first by running: python train_model.py")

def preprocess_image(image):
    """
    Preprocess an image for model prediction.
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array: Preprocessed image ready for prediction
    """
    # Convert to grayscale (handle RGBA by compositing on white)
    if image.mode in ('RGBA', 'LA'):
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image.convert('RGBA')).convert('L')
    else:
        image = image.convert('L')

    # Convert to numpy for thresholding
    arr = np.array(image)

    # Determine background from border pixels and invert to MNIST style (white digit on black)
    border = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    bg_is_white = np.mean(border) > 200
    if bg_is_white:
        # Canvas is typically black digit on white background; invert to white-on-black
        arr = 255 - arr

    # Binarize (robust to different stroke intensities)
    thresh = max(30, int(np.mean(arr) * 0.6))
    bin_arr = (arr > thresh).astype(np.uint8) * 255

    # Find bounding box of foreground
    coords = np.column_stack(np.where(bin_arr > 0))
    if coords.size == 0:
        # Fallback to simple resize/normalize if nothing detected
        small = Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS)
        x = np.array(small).astype('float32') / 255.0
        x = x[None, ..., None]
        return x

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop to bbox
    cropped = arr[y0:y1, x0:x1]

    # Make square by padding equally
    h, w = cropped.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    square = np.pad(
        cropped,
        ((pad_y, side - h - pad_y), (pad_x, side - w - pad_x)),
        mode='constant', constant_values=0
    )

    # Resize to 20x20 then pad to 28x28 (classic MNIST preprocessing)
    img_20 = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
    canvas28 = Image.new('L', (28, 28), color=0)
    canvas28.paste(img_20, (4, 4))

    # Slight blur to better match MNIST digit softness
    canvas28 = canvas28.filter(ImageFilter.GaussianBlur(radius=0.5))

    x = np.array(canvas28).astype('float32') / 255.0
    x = x[None, ..., None]
    return x

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('docs', 'index.html')

@app.route('/style.css')
def style():
    """Serve the CSS file."""
    return send_from_directory('docs', 'style.css')

@app.route('/script.js')
def script():
    """Serve the JavaScript file."""
    return send_from_directory('docs', 'script.js')

@app.route('/model/<path:filename>')
def serve_model(filename):
    """Serve TensorFlow.js model files."""
    return send_from_directory('docs/model', filename)

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
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

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
    
    # Use debug=False for production, debug=True only for development
    # For production deployment, use a production WSGI server like gunicorn
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
