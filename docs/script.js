/**
 * MNIST Digit Recognizer - Client-Side TensorFlow.js Version
 *
 * This version runs entirely in the browser using TensorFlow.js,
 * enabling deployment to static hosting like GitHub Pages.
 * Falls back to server API when running with Flask locally.
 */

// Global model reference
let model = null;
let modelLoading = false;
let useServerFallback = false;

// Detect if running on localhost (Flask server) vs static hosting (GitHub Pages)
const isLocalhost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";

// Canvas setup
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set up canvas context
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#000";

/**
 * Load the TensorFlow.js model
 */
async function loadModel() {
  if (model) return model;
  if (modelLoading) return null;
  if (useServerFallback) return null;

  modelLoading = true;
  const resultsContainer = document.getElementById("resultsContainer");

  try {
    resultsContainer.innerHTML = `
            <div class="placeholder">
                <p>⏳ Loading AI model... (first time may take a few seconds)</p>
            </div>
        `;

    // Load the model from the model/ subdirectory
    model = await tf.loadLayersModel("./model/model.json");
    console.log("Model loaded successfully (TensorFlow.js)");

    resultsContainer.innerHTML = `
            <div class="placeholder">
                <p>✅ Model loaded! Draw a digit and click "Recognize Digit"</p>
            </div>
        `;

    return model;
  } catch (error) {
    console.error("Failed to load TensorFlow.js model:", error.message);
    modelLoading = false;

    if (isLocalhost) {
      // On localhost, fall back to Flask server API
      console.log("Using server fallback for predictions");
      useServerFallback = true;
      resultsContainer.innerHTML = `
            <div class="placeholder">
                <p>✅ Ready! Draw a digit and click "Recognize Digit"</p>
                <p style="font-size: 0.8em; color: #666;">(Using server API)</p>
            </div>
        `;
    } else {
      // On GitHub Pages, show error - no server to fall back to
      resultsContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Failed to load AI model.<br>
                <small>Check browser console for details. Model path: ./model/model.json</small>
            </div>
        `;
    }

    return null;
  }
}

/**
 * Preprocess canvas image for model input
 * Converts 280x280 canvas to 28x28 grayscale tensor
 */
function preprocessCanvas() {
  return tf.tidy(() => {
    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Convert to tensor and process
    let tensor = tf.browser.fromPixels(imageData, 1); // Grayscale

    // Resize to 28x28
    tensor = tf.image.resizeBilinear(tensor, [28, 28]);

    // Invert colors (MNIST has white digits on black background)
    // Canvas has black strokes on white background
    tensor = tf.scalar(255).sub(tensor);

    // Normalize to 0-1 range
    tensor = tensor.div(255.0);

    // Add batch dimension: [1, 28, 28, 1]
    tensor = tensor.expandDims(0);

    return tensor;
  });
}

/**
 * Perform prediction on canvas content
 */
async function predict() {
  const resultsContainer = document.getElementById("resultsContainer");

  // Show loading state
  resultsContainer.innerHTML = '<div class="loading"></div>';

  try {
    // Use server fallback if TensorFlow.js model isn't available
    if (useServerFallback) {
      await predictWithServer(resultsContainer);
      return;
    }

    // Ensure model is loaded
    if (!model) {
      model = await loadModel();
      if (!model) {
        // loadModel sets useServerFallback if it fails
        if (useServerFallback) {
          await predictWithServer(resultsContainer);
        }
        return;
      }
    }

    // Preprocess the canvas
    const tensor = preprocessCanvas();

    // Run prediction
    const predictions = model.predict(tensor);
    const probabilities = await predictions.data();

    // Find the digit with highest probability
    let maxProb = 0;
    let predictedDigit = 0;
    for (let i = 0; i < 10; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        predictedDigit = i;
      }
    }

    // Format result
    const result = {
      digit: predictedDigit,
      confidence: maxProb,
      probabilities: {},
    };

    for (let i = 0; i < 10; i++) {
      result.probabilities[i.toString()] = probabilities[i];
    }

    // Clean up tensors
    tensor.dispose();
    predictions.dispose();

    // Display results
    displayResults(result);
  } catch (error) {
    console.error("Prediction error:", error);
    resultsContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Prediction failed. ${error.message}
            </div>
        `;
  }
}

/**
 * Perform prediction using server API (fallback for local development)
 */
async function predictWithServer(resultsContainer) {
  try {
    const imageData = canvas.toDataURL("image/png");

    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const result = await response.json();
    displayResults(result);
  } catch (error) {
    console.error("Server prediction error:", error);
    resultsContainer.innerHTML = `
      <div class="error-message">
        <strong>Error:</strong> Server prediction failed. ${error.message}
      </div>
    `;
  }
}

// Drawing functions
function startDrawing(e) {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  if (e.type === "mousedown") {
    [lastX, lastY] = [
      (e.clientX - rect.left) * scaleX,
      (e.clientY - rect.top) * scaleY,
    ];
  } else if (e.type === "touchstart") {
    e.preventDefault();
    const touch = e.touches[0];
    [lastX, lastY] = [
      (touch.clientX - rect.left) * scaleX,
      (touch.clientY - rect.top) * scaleY,
    ];
  }
}

function draw(e) {
  if (!isDrawing) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  let currentX, currentY;

  if (e.type === "mousemove") {
    currentX = (e.clientX - rect.left) * scaleX;
    currentY = (e.clientY - rect.top) * scaleY;
  } else if (e.type === "touchmove") {
    e.preventDefault();
    const touch = e.touches[0];
    currentX = (touch.clientX - rect.left) * scaleX;
    currentY = (touch.clientY - rect.top) * scaleY;
  }

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(currentX, currentY);
  ctx.stroke();

  [lastX, lastY] = [currentX, currentY];
}

function stopDrawing() {
  isDrawing = false;
}

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events for mobile
canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);

// Clear canvas function
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Fill with white background
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Reset results (but keep model loaded message if model is ready)
  const resultsContainer = document.getElementById("resultsContainer");
  if (model) {
    resultsContainer.innerHTML = `
            <div class="placeholder">
                <p>✅ Model ready! Draw a digit and click "Recognize Digit"</p>
            </div>
        `;
  } else {
    resultsContainer.innerHTML = `
            <div class="placeholder">
                <p>👆 Draw a digit and click "Recognize Digit" to see the prediction!</p>
            </div>
        `;
  }
}

// Initialize canvas with white background
clearCanvas();

// Clear button event
document.getElementById("clearBtn").addEventListener("click", clearCanvas);

// Predict button event
document.getElementById("predictBtn").addEventListener("click", predict);

function displayResults(result) {
  const resultsContainer = document.getElementById("resultsContainer");

  const { digit, confidence, probabilities } = result;

  // Create probability bars HTML
  let probabilityBarsHTML = "";
  for (let i = 0; i < 10; i++) {
    const prob = probabilities[i.toString()];
    const percentage = (prob * 100).toFixed(1);

    probabilityBarsHTML += `
            <div class="probability-item">
                <div class="probability-label">
                    <span class="digit-label">Digit ${i}</span>
                    <span class="probability-value">${percentage}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
  }

  // Display results
  resultsContainer.innerHTML = `
        <div class="prediction-result">
            <div class="predicted-digit">${digit}</div>
            <div class="confidence">Confidence: ${(confidence * 100).toFixed(1)}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%">
                    ${(confidence * 100).toFixed(1)}%
                </div>
            </div>
            <div class="probabilities-title">All Probabilities:</div>
            <div class="probability-bars">
                ${probabilityBarsHTML}
            </div>
        </div>
    `;
}

// Load model when page loads
window.addEventListener("load", () => {
  // Preload the model in the background
  loadModel();
});
