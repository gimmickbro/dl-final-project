import gradio as gr
import numpy as np
from PIL import Image
from skimage.util import random_noise
import cv2

def apply_noise(image, noise_type, noise_factor=0.3, mask_size=32):
    image = image.convert("RGB").resize((128, 128))
    image_np = np.array(image) / 255.0

    if noise_type == "Gaussian":
        noisy = image_np + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image_np.shape)
        noisy = np.clip(noisy, 0., 1.)

    elif noise_type == "Salt & Pepper":
        noisy = random_noise(image_np, mode='s&p', amount=0.05)

    elif noise_type == "Speckle":
        noisy = random_noise(image_np, mode='speckle')

    elif noise_type == "Poisson":
        noisy = random_noise(image_np, mode='poisson')

    elif noise_type == "Blur":
        image_255 = (image_np * 255).astype(np.uint8)
        noisy = cv2.GaussianBlur(image_255, (5, 5), sigmaX=1)
        noisy = noisy.astype(np.float32) / 255.

    elif noise_type == "Masked":
        noisy = image_np.copy()
        h, w, _ = noisy.shape
        x = np.random.randint(0, w - mask_size)
        y = np.random.randint(0, h - mask_size)
        noisy[y:y+mask_size, x:x+mask_size, :] = 0

    else:
        noisy = image_np  # default jika noise_type tidak dikenali

    return Image.fromarray((noisy * 255).astype('uint8'))

# Gradio interface
interface = gr.Interface(
    fn=apply_noise,
    inputs=[
        gr.Image(type="pil", label="Upload Gambar"),
        gr.Radio(["Gaussian", "Salt & Pepper", "Speckle", "Poisson", "Blur", "Masked"], label="Pilih Jenis Noise"),
    ],
    outputs=gr.Image(type="pil", label="Hasil Gambar Noisy"),
    title="Noise Image Generator",
    description="Upload gambar, pilih jenis noise, lalu lihat hasil gambar noisenya."
)

if __name__ == "__main__":
    interface.launch()
