from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    # Resize the image to 300x300 as expected by the model
    image = image.resize((300, 300))
    image = np.array(image)
    image = image / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)  # Ensure this is called here
        
        print(f"Processed image shape: {processed_image.shape}")  # Debug statement
        
        # Make prediction
        prediction = model.predict(processed_image)  # Ensure processed image is passed here
        
        # You can adjust this part depending on your model's output
        result = np.argmax(prediction, axis=1)[0]
        
        return jsonify({"prediction": int(result)})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
