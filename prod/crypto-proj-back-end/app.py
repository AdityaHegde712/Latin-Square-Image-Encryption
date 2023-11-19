from flask import Flask, request, jsonify, send_file
from io import BytesIO
from flask_cors import CORS
from PIL import Image
from encryption import encrypt_image

app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['image']

        # Check if the file is a valid image
        if file and allowed_file(file.filename):
            # Read the uploaded image
            uploaded_image_data = file.read()
            uploaded_image = Image.open(BytesIO(uploaded_image_data))

            processed_image = encrypt_image(uploaded_image)
            output_buffer = BytesIO()
            processed_image.save(output_buffer, format="JPEG")
            output_buffer.seek(0)

            return send_file(
                output_buffer,
                as_attachment=True,
                download_name='uploaded_image.jpg',
                mimetype='image/jpeg'
            )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decrypt', methods=["POST"])
def decrypt():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['image']

        # Check if the file is a valid image
        if file and allowed_file(file.filename):
            # Read the uploaded image
            uploaded_image = file.read()

            # You can perform any processing on the image here if needed

            # Return the uploaded image as a response
            response = BytesIO(uploaded_image)
            response.seek(0)

            return send_file(
                response,
                as_attachment=True,
                download_name='uploaded_image.jpg',
                mimetype='image/jpeg'
            )
        else:
            raise Exception()
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    # You can add more file extension checks if needed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}


if __name__ == '__main__':
    app.run(debug=True)
