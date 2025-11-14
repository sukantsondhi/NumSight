# Number-Visualizer

A machine learning web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Draw a digit on the canvas, and the AI will predict what number you drew!

## Features

- 🎨 Interactive drawing canvas for digit input
- 🤖 Deep learning model trained on MNIST dataset
- 📊 Real-time prediction with confidence scores
- 📈 Probability distribution for all digits (0-9)
- 💻 Clean and responsive web interface
- 🚀 Easy to set up and use

## Tech Stack

### Backend

- **Python 3.x**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model training
- **Flask**: Web framework for API server
- **NumPy**: Numerical computing
- **Pillow (PIL)**: Image processing

### Frontend

- **HTML5**: Structure
- **CSS3**: Styling with gradients and animations
- **JavaScript**: Canvas drawing and API interaction

### Model Architecture

- Convolutional Neural Network (CNN)
- Input: 28x28 grayscale images
- 2 Convolutional layers with MaxPooling
- Dropout layers for regularization
- Dense layers for classification
- Output: 10 classes (digits 0-9)

## Installation

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- pip package manager

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/sukantsondhi/Number-Visualizer.git
   cd Number-Visualizer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**

   ```bash
   python train_model.py
   ```

   This will:

   - Download the MNIST dataset automatically
   - Train a CNN model (takes 5-10 minutes)
   - Save the trained model to `models/mnist_model.keras` (modern format)
   - Generate a training history plot at `static/metrics/training_history.png`

4. **Run the web application**

   ```bash
   python app.py
   ```

   For development with debug mode:

   ```bash
   FLASK_DEBUG=true python app.py
   ```

### One-liner (Windows)

Prefer a single command that sets everything up? Use the helper script:

```powershell
scripts\run.ps1
```

It will create/activate a venv, install dependencies, train the model if needed, and start the app.

5. **Open your browser**
   Navigate to `http://localhost:5000`

## Production Deployment

For production deployment, it's recommended to:

1. **Disable debug mode** (default behavior)
2. **Use a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. **Use a reverse proxy** like Nginx
4. **Enable HTTPS** with SSL certificates

## Usage

1. **Draw a Digit**: Use your mouse or touchscreen to draw a digit (0-9) on the canvas
2. **Get Prediction**: Click the "Recognize Digit" button
3. **View Results**: See the predicted digit, confidence score, and probability distribution
4. **Try Again**: Click "Clear Canvas" to draw another digit

## Project Structure

```
Number-Visualizer/
├── app.py                 # Flask web server
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # Project docs
├── scripts/               # Helpers for Windows
│   └── run.ps1
├── static/                # Frontend files
│   ├── index.html         # Main HTML page
│   ├── style.css          # Styling
│   ├── script.js          # JavaScript logic
│   └── metrics/           # Generated training artifacts
│       └── training_history.png
└── models/                # Trained models (generated)
   ├── mnist_model.keras  # Preferred model format
   └── mnist_model.h5     # Optional legacy model (fallback)
```

### Model Format & Performance

The model is saved in the native Keras format (`.keras`). The server will also accept legacy HDF5 (`.h5`) if present, but `.keras` is preferred for forward compatibility.

The trained CNN model achieves:

- **Test Accuracy**: ~99%
- **Training Time**: 5-10 minutes on CPU
- **Model Size**: ~3 MB

## API Endpoints

### `GET /`

Serves the main web application

### `POST /predict`

Predicts a digit from an uploaded image

**Request Body:**

```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**

```json
{
  "digit": 7,
  "confidence": 0.9876,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0003,
    "3": 0.0004,
    "4": 0.0005,
    "5": 0.0006,
    "6": 0.0007,
    "7": 0.9876,
    "8": 0.0008,
    "9": 0.0009
  }
}
```

### `GET /health`

Health check endpoint

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## How It Works

1. **Training Phase** (`train_model.py`):

   - Loads the MNIST dataset (60k train, 10k test); normalizes to [0,1]
   - Applies light data augmentation (rotation/shift/zoom/contrast)
   - CNN with 3 conv blocks + BatchNorm, Dropout regularization
   - EarlyStopping + ModelCheckpoint save the best model (`.keras`)

2. **Inference Phase** (`app.py`):

   - User draws a digit on the canvas; image sent to Flask
   - Preprocessing centers/crops/pads to MNIST style (20x20 in 28x28),
     robustly inverts to white-on-black, binarizes, slight blur
   - Model predicts digit + probabilities returned to the frontend

3. **Frontend** (`static/`):
   - HTML5 Canvas for drawing
   - JavaScript handles drawing events
   - Sends image data to backend API
   - Displays prediction results with visualizations

## Troubleshooting

### Model not found error

- Make sure you've run `python train_model.py` first
- Check that `models/mnist_model.keras` exists (or `.h5` as fallback)

### Poor prediction accuracy

- Try drawing digits larger and centered
- Make sure the digit is dark on a light background
- Clear the canvas and try again

### Server won't start

- Check if port 5000 is already in use
- Make sure all dependencies are installed
- Verify Python version is 3.8 or higher

### TensorFlow import errors

- Ensure your virtual environment uses Python 3.10 or 3.11
- Reinstall dependencies inside the venv:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
  Then verify:
  ```powershell
  python -c "import tensorflow as tf; print(tf.__version__)"
  ```

## Security

This application implements several security best practices:

- **Debug mode disabled by default**: Prevents exposure of sensitive debugging information
- **Error message sanitization**: Stack traces are not exposed to users
- **Input validation**: Images are validated before processing
- **Safe dependencies**: All dependencies are regularly updated for security

For production deployment, additional security measures should be implemented:

- Use HTTPS/SSL encryption
- Implement rate limiting
- Add authentication if needed
- Use environment variables for configuration
- Regular security audits and updates

## PR Readiness Checklist

Use this checklist when opening a pull request:

- App runs locally: `python app.py` serves UI and `/predict` works
- Model present or reproducible: `python train_model.py` succeeds
- README updated with any changes to setup or run
- No large, unused artifacts committed (e.g., datasets, temporary files)
- Code adheres to project style and keeps changes minimal and focused

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, and Christopher Burges
- TensorFlow/Keras team for the excellent deep learning framework
- Flask team for the web framework

## Future Enhancements

- [ ] Support for multiple digit recognition
- [ ] Model fine-tuning options
- [ ] Export predictions to file
- [ ] Mobile app version
- [ ] Real-time drawing predictions
- [ ] User feedback collection for model improvement
