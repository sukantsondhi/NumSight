"""
MNIST Digit Recognition Model Training Script

This script trains a Convolutional Neural Network (CNN) on the MNIST dataset
to recognize handwritten digits (0-9).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

def create_model():
    """
    Create a CNN model for MNIST digit classification.
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Conv block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv block 3 (small)
        layers.Conv2D(96, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    try:
        # Try to load MNIST dataset
        print("Attempting to download MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Warning: Could not download MNIST dataset: {e}")
        print("Creating a synthetic dataset for demonstration...")
        
        # Create a simple synthetic dataset with basic digit patterns
        # This is just for demonstration and will have lower accuracy
        np.random.seed(42)
        
        # Generate 6000 training samples and 1000 test samples
        n_train = 6000
        n_test = 1000
        
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        # Create simple patterns for each digit
        for i in range(n_train):
            digit = i % 10
            img = create_digit_pattern(digit)
            x_train.append(img)
            y_train.append(digit)
        
        for i in range(n_test):
            digit = i % 10
            img = create_digit_pattern(digit)
            x_test.append(img)
            y_test.append(digit)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        print("Synthetic dataset created successfully!")
        print("Note: For production use, please ensure internet access to download the real MNIST dataset.")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

def create_digit_pattern(digit):
    """
    Create a simple synthetic pattern for a digit.
    
    Args:
        digit: Integer from 0-9
        
    Returns:
        28x28 numpy array with a simple pattern
    """
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # Add some noise
    noise = np.random.randint(0, 30, (28, 28), dtype=np.uint8)
    img += noise
    
    # Create simple patterns for each digit
    if digit == 0:
        # Circle
        for i in range(6, 22):
            for j in range(6, 22):
                if 36 < (i-14)**2 + (j-14)**2 < 64:
                    img[i, j] = 200 + np.random.randint(0, 55)
    
    elif digit == 1:
        # Vertical line
        for i in range(6, 22):
            img[i, 13:15] = 200 + np.random.randint(0, 55)
    
    elif digit == 2:
        # Top horizontal, diagonal, bottom horizontal
        img[8, 8:20] = 200 + np.random.randint(0, 55)
        for i in range(8, 20):
            img[i, 20-i+8] = 200 + np.random.randint(0, 55)
        img[19, 8:20] = 200 + np.random.randint(0, 55)
    
    elif digit == 3:
        # Three horizontal lines on right
        img[8, 10:20] = 200 + np.random.randint(0, 55)
        img[14, 10:20] = 200 + np.random.randint(0, 55)
        img[20, 10:20] = 200 + np.random.randint(0, 55)
        for i in range(8, 21):
            img[i, 19] = 200 + np.random.randint(0, 55)
    
    elif digit == 4:
        # Vertical line on right, horizontal in middle, diagonal
        for i in range(6, 22):
            img[i, 17] = 200 + np.random.randint(0, 55)
        img[14, 8:20] = 200 + np.random.randint(0, 55)
        for i in range(6, 15):
            img[i, 8] = 200 + np.random.randint(0, 55)
    
    elif digit == 5:
        # Top, left, middle, bottom right
        img[7, 9:20] = 200 + np.random.randint(0, 55)
        for i in range(7, 14):
            img[i, 9] = 200 + np.random.randint(0, 55)
        img[14, 9:20] = 200 + np.random.randint(0, 55)
        for i in range(14, 21):
            img[i, 19] = 200 + np.random.randint(0, 55)
        img[20, 9:20] = 200 + np.random.randint(0, 55)
    
    elif digit == 6:
        # Circle with gap on top right
        for i in range(7, 21):
            for j in range(9, 19):
                if 25 < (i-14)**2 + (j-14)**2 < 49:
                    img[i, j] = 200 + np.random.randint(0, 55)
        for i in range(7, 21):
            img[i, 9] = 200 + np.random.randint(0, 55)
    
    elif digit == 7:
        # Top horizontal and diagonal
        img[7, 9:20] = 200 + np.random.randint(0, 55)
        for i in range(7, 21):
            j = 20 - int((i - 7) * 0.7)
            if 9 <= j < 20:
                img[i, j] = 200 + np.random.randint(0, 55)
    
    elif digit == 8:
        # Two circles
        for i in range(7, 21):
            for j in range(9, 19):
                dist_top = (i-10)**2 + (j-14)**2
                dist_bot = (i-18)**2 + (j-14)**2
                if (16 < dist_top < 36) or (16 < dist_bot < 36):
                    img[i, j] = 200 + np.random.randint(0, 55)
    
    elif digit == 9:
        # Circle with gap on bottom left
        for i in range(7, 21):
            for j in range(9, 19):
                if 25 < (i-14)**2 + (j-14)**2 < 49:
                    img[i, j] = 200 + np.random.randint(0, 55)
        for i in range(7, 21):
            img[i, 19] = 200 + np.random.randint(0, 55)
    
    return img

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Keras training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # Ensure metrics dir exists (so frontend can serve the plot if desired)
    os.makedirs(os.path.join('static', 'metrics'), exist_ok=True)
    plot_path = os.path.join('static', 'metrics', 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to '{plot_path}'")

def main():
    """Main training function."""
    print("=" * 60)
    print("MNIST Digit Recognition - Model Training")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing MNIST dataset...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Create model
    print("\nCreating CNN model...")
    model = create_model()
    model.summary()
    
    # Train model
    # Data augmentation to better match canvas drawings
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])

    # Build augmented dataset
    aug_x = data_augmentation(x_train)

    # Callbacks: early stopping and checkpoint best model
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join('models', 'mnist_model.keras'),
                                        monitor='val_accuracy',
                                        save_best_only=True)
    ]

    print("\nTraining model with augmentation and callbacks...")
    history = model.fit(
        aug_x, y_train,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Ensure best model is saved by checkpoint; for safety, save final weights too
    model_path = os.path.join('models', 'mnist_model.keras')
    try:
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    except Exception as e:
        print(f"Warning: Could not save final model due to: {e}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
