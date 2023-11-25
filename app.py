from flask import Flask , render_template , request, send_from_directory, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2


app =Flask(__name__)

model = load_model('train_2.h5' ,compile =False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if 'image' in request.files:
            
            f = request.files['image']
            if f.filename != '':
                # Specify the directory to save uploaded images
                upload_folder = 'uploads'

                # Ensure the upload directory exists; if not, create it
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)

                # Save the uploaded image to the specified directory
                image_path = os.path.join(upload_folder, f.filename)
                f.save(image_path)

                img = image.load_img(image_path, target_size= (232,232))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the uploaded image to gray-scale
                x = image.img_to_array(img)
                print(x)
                x=np.expand_dims(x, axis=0)
                print(x)
                y = model.predict(x)
                preds = np.argmax(y, axis=1)

                print('prediction', preds)

                index = ['COVID','NORMAL', 'PNEUMONIA']
                if preds[0]==0:
                    statement = "The person is predicted to have symptoms of "
                elif preds[0]==1:
                    statement = "The Person has no symptoms of any disease and is predicted as "
                else:
                    statement="The person is predicted to have symptoms of "

                text = statement + str(index[preds[0]])
                print(str(index[preds[0]]))
    
     

                # Pass the image path to the 'result.html' template for display
                return render_template('result.html', image_path=f.filename, text=text)

    return render_template('upload.html')

@app.route('/upload/<path:filename>')
def get_image(filename):
    return send_from_directory('uploads', filename)

@app.route('/result' , methods=['POST'])
def result():
    image_path = request.args.get('image_path')
    return render_template('result.html', image_path=image_path)




if __name__ == '__main__':
    app.run(debug=True)
    