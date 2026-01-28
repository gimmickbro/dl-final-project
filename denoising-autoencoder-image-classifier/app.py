import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
autoencoder = load_model('autoencoder_denoise.h5')
classifier = load_model('tomato_cnn_model.h5')

# Mapping index ke nama kelas
label_mapping = {0: 'Reject', 1: 'Ripe', 2: 'Unripe'}

def process_image(image):
    # Preprocessing
    image = image.convert("RGB").resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Denoising
    denoised_img = autoencoder.predict(img_array)

    # Classification
    prediction = classifier.predict(denoised_img)[0]

    # Convert denoised image back to displayable
    denoised_img_disp = np.squeeze(denoised_img) * 255
    denoised_img_disp = Image.fromarray(denoised_img_disp.astype('uint8'))

    # Confidence dict: {label_name: float_confidence}
    confidences = {label_mapping[i]: float(pred) for i, pred in enumerate(prediction)}

    return denoised_img_disp, confidences

# Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Denoised Image"),
        gr.Label(label="Predicted Class & Confidence")
    ],
    title="Denoising Autoencoder + Image Classifier",
    description="Upload gambar noise, hasilnya gambar bersih & hasil klasifikasinya."
)

# Run Gradio
if __name__ == "__main__":
    interface.launch()
