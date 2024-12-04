from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import traceback  # For detailed error logging
import os  # For file and directory operations
import time

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for specific routes (replace 'http://localhost:3001' with your frontend domain)
CORS(app, resources={r"/process-image": {"origins": ["http://localhost:3000", "http://localhost:3001"]}})

# Load the Keras model
MODEL_PATH = 'generator_final.keras'
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}.")
except Exception as e:
    print(f"Error loading the model from {MODEL_PATH}: {e}")
    traceback.print_exc()
    model = None  # Set model to None to handle issues gracefully

# Directory for saving generated images
GENERATED_DIR = 'generated'
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Endpoint to process an image using the Keras model.
    """
    try:
        # Ensure an image file is provided in the request
        if 'imageInput' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['imageInput']
        print(f"Image uploaded: {image_file.filename}")

        # Validate the file type
        if not image_file.mimetype.startswith('image/'):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_file.read()))
        print(f"Original image size: {image.size}")
        image = image.convert('RGB').resize((256, 256))  # Ensure RGB and 256x256
        print(f"Resized image to: {image.size}")

        # Normalize the image and reshape for the model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        print(f"Image array shape for model input: {image_array.shape}")

        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 500

        # Generate the image using the model
        print("Generating image using the model...")
        generated_image = model.predict(image_array)
        print("Image generation completed.")

        # Post-process the generated image
        generated_image = (generated_image[0] * 255).astype(np.uint8)
        if generated_image.shape[-1] == 1:  # Grayscale to RGB
            generated_image = np.squeeze(generated_image, axis=-1)

        # Convert the generated image to a PIL Image
        generated_image_pil = Image.fromarray(generated_image)

        # Save the generated image with a unique filename
        output_filename = os.path.join(GENERATED_DIR, f"generated_{int(time.time())}.png")
        generated_image_pil.save(output_filename)
        print(f"Generated image saved at: {output_filename}")

        # Convert the image to a byte stream for response
        img_byte_arr = io.BytesIO()
        generated_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the generated image as binary data
        return img_byte_arr.getvalue(), 200, {'Content-Type': 'image/png'}

    except Exception as e:
        # Log detailed error and traceback
        print("Error occurred while processing image:", e)
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred while processing the image. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
