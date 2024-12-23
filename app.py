from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from io import BytesIO
import os
import asyncio



# Initialize Flask app
app = Flask(__name__)

gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)]
            )

# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.getcwd(),"BrainTumor.h5"))

converter= tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model= converter.convert()
with open('converted_model.tflite', 'wb') as f:     
  f.write(tflite_model)

interpreter= tf.lite.Interpreter(os.path.join(os.getcwd(), 'converted_model.tflite'))
interpreter.allocate_tensors()



print(os.path.join(os.getcwd(),"BrainTumor.h5"))

# Route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            print('Got here:', request)
            # Check if a file is uploaded
            if "file" not in request.files:
                return render_template("home.html", result="No file uploaded!")

            file = request.files["file"]
            if file.filename == "":
                return render_template("home.html", result="No selected file!")
            
            # img = BytesIO(file.read())
            #if not os.path.exists('tmp'):
                #os.mkdir('tmp')

            img_path= file.filename
            full_path= os.path.join('tmp', img_path)
            file.save(full_path)    
            print(full_path)



            # Process the uploaded image
            loaded_img = tf.keras.utils.load_img(full_path, target_size=(128, 128))
            img_to_array = tf.keras.utils.img_to_array(loaded_img) / 255.0
            img_expanded = np.expand_dims(img_to_array, axis=0)

            

            input_details= converter.get_input_details()
            output_details= converter.get_output_details()

            interpreter.get_tensors(input_details[0]['index'], img_expanded)

            interpreter.invoke()




            # Make prediction
            predictions =  interpreter.get_tensors(output_details[0]['index'])
            print(predictions)  # For debugging purposes
            result = "Healthy" if predictions[0][0] > 0.5 else "Tumor"

            os.remove(full_path)

            # Return result to the webpage
            return render_template("home.html", result= result)
        except:
            print('Something went wrong')
            return render_template("home.html", result=None)
        # return render_template("home.html", result=None)
    return render_template("home.html", result=None)

# Main function to run the Flask app
if __name__ == "__main__":
    app.run()
