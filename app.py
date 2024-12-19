from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from io import BytesIO
import os


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.getcwd(),"BrainTumor.h5"))

# Route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Check if a file is uploaded
        if "file" not in request.files:
            return render_template("home.html", result="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("home.html", result="No selected file!")
        
        img = BytesIO(file.read())

        # Process the uploaded image
        loaded_img = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
        img_to_array = tf.keras.preprocessing.image.img_to_array(loaded_img) / 255.0
        img_expanded = np.expand_dims(img_to_array, axis=0)

        # Make prediction
        predictions = model.predict(img_expanded)
        print(predictions)  # For debugging purposes
        result = "Healthy" if predictions[0][0] > 0.5 else "Tumor"

        # Return result to the webpage
        return render_template("home.html", result=result)

    return render_template("home.html", result=None)


# Main function to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
