from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os




# Initialize Flask app
app = Flask(__name__)



# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.getcwd(),"BrainTumor.h5"))

converter= tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model= converter.convert()

if not os.path.exists(os.path.join(os.getcwd(), "converted_model.tflite")):
    with open('converted_model.tflite', 'wb') as f:     
       f.write(tflite_model)

interpreter= tf.lite.Interpreter(os.path.join(os.getcwd(), 'converted_model.tflite'))
interpreter.allocate_tensors()



# Route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            print('Got here:', request)

            file = request.files["file"]
            if file.filename == "":
                return render_template("home.html", result="No selected file!")


            img_path= file.filename
            full_path= os.path.join('tmp', img_path)
            file.save(full_path)    
            print(full_path)

            print('got here now')



            # Process the uploaded image
            loaded_img = tf.keras.utils.load_img(full_path, target_size=(128, 128))
            img_to_array = tf.keras.utils.img_to_array(loaded_img) / 255.0
            img_expanded = np.expand_dims(img_to_array, axis=0)
            print(type(img_expanded))

            print('got here too')

            input_details= interpreter.get_input_details()
            output_details= interpreter.get_output_details()

            print('got here 3')
            
            interpreter.set_tensor(input_details[0]['index'], img_expanded)

            interpreter.invoke()

            print('got here 4')
            # Make prediction
            predictions = interpreter.get_tensor(output_details[0]['index'])
            print(predictions)  # For debugging purposes
            result = "Healthy" if predictions[0][0] > 0.5 else "Tumor"
            print('got here 5')

            os.remove(full_path)
            print('got here 6')

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