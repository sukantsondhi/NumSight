"""
Convert Keras MNIST model to TensorFlow.js format for browser-based inference.

This script converts the trained Keras model to TensorFlow.js format,
enabling client-side predictions on GitHub Pages without a backend server.

Usage:
    python convert_to_tfjs.py

Requirements:
    pip install -r requirements-tfjs.txt
    
    Or if you encounter dependency issues:
    pip install tensorflowjs==4.17.0 --no-deps
    pip install tensorflow-hub tf-keras h5py jax jaxlib flax importlib_resources --no-deps
    pip install "setuptools<70"
"""

import os
import shutil


def convert_model():
    """Convert the Keras model to TensorFlow.js format."""
    
    # Check for tensorflowjs
    try:
        import tensorflowjs as tfjs
        import tensorflow as tf
    except ImportError as e:
        print("Error: tensorflowjs not installed or has missing dependencies.")
        print(f"Details: {e}")
        print("\nTo fix, run:")
        print("  pip install -r requirements-tfjs.txt")
        print("\nOr install manually:")
        print("  pip install tensorflowjs==4.17.0 --no-deps")
        print("  pip install tensorflow-hub tf-keras h5py jax jaxlib flax importlib_resources --no-deps")
        print('  pip install "setuptools<70"')
        return False
    
    # Model paths
    model_dir = 'models'
    keras_model_path = os.path.join(model_dir, 'mnist_model.keras')
    h5_model_path = os.path.join(model_dir, 'mnist_model.h5')
    output_dir = os.path.join('docs', 'model')
    
    # Find the model file
    model_path = None
    if os.path.exists(keras_model_path):
        model_path = keras_model_path
    elif os.path.exists(h5_model_path):
        model_path = h5_model_path
    else:
        print(f"Error: No model found in '{model_dir}/'")
        print("Please train the model first: python train_model.py")
        return False
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Convert to TensorFlow.js format
    print("Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, output_dir)
    
    print(f"\nSuccess! Model saved to: {output_dir}/")
    print("\nFiles created:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({size:,} bytes)")
    
    return True


if __name__ == '__main__':
    convert_model()
