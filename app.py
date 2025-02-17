import os

from flask import Flask, request, send_file, render_template
from gradio_client import Client, handle_file
from PIL import Image
from flask import jsonify
import io



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
client = Client("franciszzj/Leffa")

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def merge_images():
    person_img = request.files.get('my_image')
    apparel_img = request.files.get('clothes_image')

    print(person_img)
    print(apparel_img)

    if not person_img or not apparel_img:
        return "Both images are required", 400

    try:
        # Save the uploaded images to the uploads folder
        person_path = os.path.join(UPLOAD_FOLDER, person_img.filename)
        apparel_path = os.path.join(UPLOAD_FOLDER, apparel_img.filename)


        person_img.save(person_path)
        apparel_img.save(apparel_path)
        print('Image saved')
        result = client.predict(
            src_image_path=handle_file(person_path),  # Comma added
            ref_image_path=handle_file(apparel_path),  # Comma added
            ref_acceleration=False,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            api_name="/leffa_predict_vt"
        )
        print(result)
        # Step 1: Grab the first item
        first_image_path = result[0]
        # Step 2: Convert the .webp image to .jpeg
        with Image.open(first_image_path) as img:
            jpeg_image_path = os.path.join(UPLOAD_FOLDER, 'converted_image.jpeg')
            img.convert('RGB').save(jpeg_image_path, 'JPEG')
        return send_file(jpeg_image_path, mimetype="image/jpeg")

    except Exception as e:
        return f"Error processing images: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
