from base64 import encodebytes
import zipfile
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from flask_cors import CORS
import cv2
from PIL import Image
from encryption import encrypt_image
import numpy as np
from mainBackend2 import encryptTotal

app = Flask(__name__)
CORS(app)

LE = None


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['image']

        # Check if the file is a valid image
        if file and allowed_file(file.filename):

            global LE
            # Read the uploaded image
            uploaded_image_data = file.read()
            uploaded_image = Image.open(BytesIO(uploaded_image_data))
            grayscale_image = uploaded_image.convert('L')
            grayscale_image = grayscale_image.resize([256, 256])

            enc, dec = encryptTotal(np.array(grayscale_image))
            processed_image_dec = Image.fromarray(dec)
            processed_image_enc = Image.fromarray(enc)
            processed_image_dec.save(
                "buffer/processed_img_dec.png", format="PNG")
            processed_image_enc.save(
                "buffer/processed_img_enc.png", format='PNG')
            zipf = zipfile.ZipFile('something.zip', 'w', zipfile.ZIP_DEFLATED)
            zipf.write('buffer/processed_img_dec.png')
            zipf.write('buffer/processed_img_enc.png')
            zipf.close()

            return send_file(
                'something.zip',
                as_attachment=True,
                download_name='something.zip',
                mimetype='application/zip'
            )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# @app.route('/decrypt', methods=["POST"])
# def decrypt():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['image']

#         # Check if the file is a valid image
#         if file and allowed_file(file.filename):
#             # Read the uploaded image
#             uploaded_image_data = file.read()
#             uploaded_image = Image.open(BytesIO(uploaded_image_data))
#             uploaded_image.save('buffer/'+request.files['image'].filename)

#             print(type(LE))

#             processed_image = decryptTotal(LE,
#                                            'buffer/'+request.files['image'].filename)
#             processed_image = Image.fromarray(processed_image)
#             output_buffer = BytesIO()
#             processed_image.save(output_buffer, format="PNG")
#             output_buffer.seek(0)

#             return send_file(
#                 output_buffer,
#                 as_attachment=True,
#                 download_name='uploaded_image.jpg',
#                 mimetype='image/jpeg'
#             )
#         else:
#             raise Exception()
#     except Exception as e:
#         return jsonify({'error': str(e.with_traceback())}), 500


def allowed_file(filename):
    # You can add more file extension checks if needed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}


if __name__ == '__main__':
    app.run(debug=True)
