from logging import exception
import os

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import model_functions
from azure.storage.blob import BlobServiceClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Allow requests only from localhost:3000
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/api/*": {"origins": "https://wowdaoai.blob.core.windows.net"}})




connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING') # retrieve the connection string from the environment variable
container_name = "photos" # container name in which images will be store in the storage account

blob_service_client = BlobServiceClient.from_connection_string(conn_str='DefaultEndpointsProtocol=https;AccountName=wowdaoai;AccountKey=lRFWGShUj1PwEIXhyrvJAGWBAU4v5rKyCuZquthK5DI2XU9XmOOE2Cs8qrR2pRAa6c8xezXxC8nP+ASt6+edlw==;EndpointSuffix=core.windows.net') # create a blob service client to interact with the storage account
try:
    container_client = blob_service_client.get_container_client(container=container_name) # get container client to interact with the container in which images will be stored
    container_client.get_container_properties() # get properties of the container to force exception to be thrown if container does not exist
except Exception as e:
    print(e)
    print("Creating container...")
    container_client = blob_service_client.create_container(container_name) # create a container in the storage account if it does not exist



@app.route('/', methods=['GET'])
def func():
    return "Welcome to Root server"




@app.route('/api/v1/predict', methods=['POST'])
def predict():
    print("Predicting-------->>> Please Wait")

    try:
         uploaded_file = request.files['image']
         upload_folder = "./Input/"

         if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

         if uploaded_file.filename != '':
                filename = uploaded_file.filename
                file_path = os.path.join(upload_folder, filename)
                uploaded_file.save(file_path)
                
                print("File Saved into the ./Input/ folder")
 
    except exception as e:
         return jsonify({"error": "No image provided"})


    # Checking and loading different models based upon the type
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    conversion_type = request.args.get('type')

    if(conversion_type == "mri-ct"):
        print("I'm currently performing the MRI-CT prediction")
        reconstructed_model=tf.keras.models.load_model("./model.keras")

    elif(conversion_type == "ct-mri"):
        print("I'm currently performing the CT-MRI prediction")
        reconstructed_model=tf.keras.models.load_model("./model1.keras")

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    

    test_dataset = tf.data.Dataset.list_files("./Input/*.jpg")
    test_dataset = test_dataset.map(model_functions.load_image_test)
    test_dataset = test_dataset.batch(model_functions.BATCH_SIZE)


# Reading the Index value from the text file
    with open('./index.txt', 'r') as file:
        index = int(file.read())

    for example_input in test_dataset.take(1):
            
            model_functions.generate_images(reconstructed_model, example_input,1)
            
            container_client.upload_blob(name=container_name+str(index) , data=open("./predictions/predicted1.jpg", "rb").read()) # upload the image to the container in the storage account            
            
            blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{container_name+str(index)}"
            index=index+1
    print("Prediction is successfully saveed in the Azure-blob-storage")
    
    # Saving the state of Index value in a text file
    with open('./index.txt', 'w') as file:
        file.write(str(index))
    

    return jsonify({"blob_url" : blob_url})

def new_func():
    return request.form.get

if __name__ == '__main__':
    app.run(debug=True)