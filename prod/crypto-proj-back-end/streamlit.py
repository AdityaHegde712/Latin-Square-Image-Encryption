import streamlit as st
from PIL import Image
import numpy as np
from mainBackend2 import encryptTotal

# Function to convert image to grayscale and resize


def process_image(input_image):
    # Convert the image to grayscale
    gray_image = input_image.convert('L')

    # Resize the image to 256x256
    resized_image = gray_image.resize((256, 256))

    return resized_image

# Function that takes an image as a numpy array and returns two images as numpy arrays


# Streamlit application


def main():
    st.title("Image Processing App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the original image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image",
                 use_column_width=True)

        # Process the image
        processed_image = process_image(original_image)

        # Convert processed image to numpy array
        processed_array = np.array(processed_image)

        # Process the numpy array
        result_array_1, result_array_2 = encryptTotal(processed_array)

        # Display the results
        st.image(result_array_1, caption="Encrypted Image",
                 use_column_width=True)
        original_dimensions = original_image.size
        resized_result_image_2 = Image.fromarray(
            result_array_2).resize(original_dimensions)

        st.image(resized_result_image_2,
                 caption="Decrypted Image", use_column_width=True)


if __name__ == "__main__":
    main()
