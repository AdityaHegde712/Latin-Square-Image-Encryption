<script setup>
import { ref } from "vue";
import constants from "../constants";

const encryptedImageReady = ref(false);
const encryptedImageURL = ref(null);
const encryptedImageBlob = ref(null);

const decryptedImageReady = ref(false);
const decryptedImageURL = ref(null);

const uploadedImage = ref(null);
const uploadedImageURL = ref(null);

const uploadImage = async () => {
  uploadedImageURL.value = URL.createObjectURL(uploadedImage.value.files[0]);

  const form = new FormData();
  form.append("image", uploadedImage.value.files[0]);

  const response = await fetch(constants.BASE_URL + "/upload", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    console.log(await response.json());
  } else {
    encryptedImageBlob.value = await response.blob();

    encryptedImageURL.value = URL.createObjectURL(encryptedImageBlob.value);
    encryptedImageReady.value = true;

    uploadEncrypted(encryptedImageReady.value);
  }
};

const uploadEncrypted = async (encryptedImageReady) => {
  if (!encryptedImageReady) return;

  const form = new FormData();
  // const res = await fetch(encryptedImageURL);
  const data = new File([encryptedImageBlob.value], "encrypted_im.png");
  form.append("image", data);

  console.log(data);

  const response = await fetch(constants.BASE_URL + "/decrypt", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    console.log(await response.json());
  } else {
    const blob = await response.blob();

    console.log(blob);

    decryptedImageURL.value = URL.createObjectURL(blob);
    decryptedImageReady.value = true;
  }
};
</script>
<template>
  <div
    class="min-h-[100vh] bg-slate-100 flex items-center justify-center mt-4 pb-20"
  >
    <div class="bg-white rounded-lg p-8 space-y-8 w-full max-w-md min-w-[50%]">
      <h1 class="text-2xl font-bold text-center">Image Upload and Display</h1>

      <form @submit.prevent="uploadImage" class="space-y-4">
        <div class="mb-4">
          <label
            for="imageUpload"
            class="block text-sm font-medium text-gray-700"
          >
            Upload an Image
          </label>
          <input
            type="file"
            id="imageUpload"
            ref="uploadedImage"
            accept="image/*"
            class="mt-1 py-2 px-3 block w-full rounded-md border focus:ring-indigo-500 focus:border-indigo-500 text-gray-700"
          />
        </div>

        <button
          type="submit"
          class="bg-indigo-500 text-white font-semibold py-2 px-4 rounded-lg w-full"
        >
          Upload
        </button>
      </form>

      <div v-if="!!uploadedImageURL">
        <h2 class="text-xl font-semibold text-center">Uploaded Image</h2>
        <img
          :src="uploadedImageURL"
          alt="Uploaded Image"
          class="w-full max-w-xl mx-auto"
        />
      </div>

      <div v-if="!!encryptedImageReady">
        <h2 class="text-xl font-semibold text-center">Encrypted Image</h2>
        <img
          :src="encryptedImageURL"
          alt="Encrypted Image"
          class="w-full max-w-xl mx-auto"
        />
      </div>
      <div v-if="!!decryptedImageReady">
        <h2 class="text-xl font-semibold text-center">Decrypted Image</h2>
        <img
          :src="decryptedImageURL"
          alt="Encrypted Image"
          class="w-full max-w-xl mx-auto"
        />
      </div>
    </div>
  </div>
</template>
