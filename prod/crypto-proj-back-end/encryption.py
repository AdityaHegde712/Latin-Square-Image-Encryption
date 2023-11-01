import numpy as np
from PIL import Image
np.random.seed(42)

def generate_latin_square(size):
    latin_square = np.arange(size)
    np.random.shuffle(latin_square)
    return latin_square

class LatinSquareEncryption:
    def __init__(self, image_data, secret_key):
        self.image_data = image_data
        self.secret_key = secret_key
        self.s_box = self.generate_s_box()

    def generate_s_box(self):
        s_box = np.arange(256)
        target_seed = int.from_bytes(self.secret_key, "big") // (2 ** 224)
        np.random.seed(target_seed)
        np.random.shuffle(s_box)
        return s_box

    def generate_latin_square(self):
        self.latin_square = np.arange(self.image_data.size)
        np.random.shuffle(self.latin_square)

    def S_FRM(self, prev_ciphertext, plaintext):
        # Use the S-box to transform the plaintext into ciphertext
        return self.s_box[plaintext ^ prev_ciphertext]

    def S_IRM(self, prev_ciphertext, ciphertext):
        # Find the index of the ciphertext in the S-box
        return np.where(self.s_box == ciphertext)[0][0] ^ prev_ciphertext

    def S_FCM(self, prev_ciphertext, plaintext):
        # Similar to FRM but operates on columns instead of rows
        return self.s_box[plaintext ^ prev_ciphertext]

    def S_ICM(self, prev_ciphertext, ciphertext):
        # Similar to IRM but operates on columns instead of rows
        return np.where(self.s_box == ciphertext)[0][0] ^ prev_ciphertext

    # Step 2: Latin Square Whitening
    def latin_square_whitening(self):
        whitened_data = np.copy(self.image_data)
        for i in range(self.image_data.size):
            whitened_data.flat[i] = (self.image_data.flat[i] ^ self.latin_square[i]) % 256
        return whitened_data
    
    def inverse_latin_square_whitening(self, whitened_image_data):
        self.data = np.copy(whitened_image_data)
        for i in range(whitened_image_data.size):
            self.data.flat[i] = (whitened_image_data.flat[i] ^ self.latin_square[i]) % 256
        return self.data

    # Step 3: Latin Square Substitution
    def latin_square_substitution(self, whitened_data):
        # Apply Latin square substitution
        substituted_data = np.copy(whitened_data)

        # Insert FRM and FCM here
        for r in range(whitened_data.shape[0]):
            for c in range(whitened_data.shape[1]):
                if r != 0: substituted_data[r,c] = self.S_FRM(substituted_data[r-1,c], whitened_data[r,c])
                else: substituted_data[r,c] = self.S_FRM(0, whitened_data[r,c])

                if c != 0: substituted_data[r,c] = self.S_FCM(substituted_data[r,c-1], whitened_data[r,c])
                else: substituted_data[r,c] = self.S_FCM(0, whitened_data[r,c])

        # Return substituted_data
        return substituted_data
    
    def inverse_latin_square_substitution(self, substituted_data):
        pass
    
    # Step 4: Substituted data permutation
    def latin_square_permutation(self, substituted_data):
        # Initialize an empty array for the permuted_data
        permuted_data = np.copy(substituted_data)

        # Apply the Forward Row/Column Mapping functions
        for i in range(1, substituted_data.size):
            permuted_data.flat[i] = self.S_FRM(permuted_data.flat[i-1], substituted_data.flat[i])
            permuted_data.flat[i] = self.S_FCM(permuted_data.flat[i-substituted_data.shape[1]], permuted_data.flat[i])

        return permuted_data
    
    def inverse_latin_square_permutation(self, permuted_data):
        pass

    def get_encrypted_image(self):
        self.encrypted_image = self.permuted_data.reshape(self.image_data.shape)
        return self.encrypted_image


# Main program
if __name__ == "__main__":
    # Load an example image (replace with your image loading code)
    image_path = "test_images\\aa_gray.jpg"  # Replace with the path to your image file
    image = Image.open(image_path).convert('L')

    # Convert the image to a numpy array
    image_data = np.array(image)

    # Step 1: Generate a secret key K (32 bytes)
    secret_key = np.random.bytes(32)

    crypto = LatinSquareEncryption(image_data, secret_key)

    # Generate a Latin square (substitution matrix) for the given size
    crypto.generate_latin_square()

    # Step 2: Latin Square Whitening
    whitened_image_data = crypto.latin_square_whitening()

    # # Step 3: Latin Square Substitution
    substituted_image_data = crypto.latin_square_substitution(whitened_image_data)

    # Step 4: Latin Square Permutation
    encrypted_image_data = crypto.latin_square_permutation(substituted_image_data)

    # # Decryption
    # # Step 5: Inverse Latin Square Permutation
    # inverse_permuted_data = crypto.inverse_latin_square_permutation(encrypted_image_data)

    # # Step 6: Inverse Latin Square Substitution
    # inverse_substituted_data = crypto.inverse_latin_square_substitution(inverse_permuted_data)

    # # Inverse Whitening
    # image_data = crypto.inverse_latin_square_whitening(inverse_substituted_data)

    # Display the encrypted image
    whitened_image = Image.fromarray(whitened_image_data, 'L')
    reg_image = Image.fromarray(image_data, 'L')
    substituted_image = Image.fromarray(substituted_image_data, 'L')
    encrypted_image = Image.fromarray(encrypted_image_data, 'L')

    # original_image = Image.fromarray(image_data, 'L')
    # original_image.save("decrypted_images\\aa_gray_original.jpg")
    

    # Alternatively, you can save the encrypted image to a file
    whitened_image.save("encrypted_images\\aa_whitened.jpg")
    substituted_image.save("encrypted_images\\aa_substituted.jpg")
    encrypted_image.save("encrypted_images\\aa_permuted_(encrypted).jpg")