// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set up canvas context
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000';

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.type === 'mousedown') {
        [lastX, lastY] = [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
    } else if (e.type === 'touchstart') {
        e.preventDefault();
        const touch = e.touches[0];
        [lastX, lastY] = [(touch.clientX - rect.left) * scaleX, (touch.clientY - rect.top) * scaleY];
    }
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    let currentX, currentY;
    
    if (e.type === 'mousemove') {
        currentX = (e.clientX - rect.left) * scaleX;
        currentY = (e.clientY - rect.top) * scaleY;
    } else if (e.type === 'touchmove') {
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
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Clear canvas function
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Fill with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Reset results
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = `
        <div class="placeholder">
            <p>👆 Draw a digit and click "Recognize Digit" to see the prediction!</p>
        </div>
    `;
}

// Initialize canvas with white background
clearCanvas();

// Clear button event
document.getElementById('clearBtn').addEventListener('click', clearCanvas);

// Predict button event
document.getElementById('predictBtn').addEventListener('click', async () => {
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Show loading state
    resultsContainer.innerHTML = '<div class="loading"></div>';
    
    try {
        // Get canvas image data
        const imageData = canvas.toDataURL('image/png');
        
        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        resultsContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Failed to get prediction. Make sure the server is running and the model is trained.
            </div>
        `;
    }
});

function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    const { digit, confidence, probabilities } = result;
    
    // Create probability bars HTML
    let probabilityBarsHTML = '';
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

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (!health.model_loaded) {
            console.warn('Model not loaded on server');
        }
    } catch (error) {
        console.error('Server health check failed:', error);
    }
});
