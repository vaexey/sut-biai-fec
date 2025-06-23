from tensorflow.keras.models import load_model
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image

# --- ARGPARSE ---
parser = argparse.ArgumentParser(description="Test Keras model on a single grayscale image")
parser.add_argument('--model', type=str, required=True, help='Path to .keras model file')
parser.add_argument('--image', type=str, required=True, help='Path to test image')
args = parser.parse_args()

# --- CONFIG ---
target_size = (48, 48)  # For FER-style grayscale images
batch_size = 32

classes = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
# --- LOAD MODEL ---
model = load_model(args.model)

img = image.load_img(args.image, color_mode='grayscale', target_size=(48, 48))

# Convert to numpy array
img_array = image.img_to_array(img)

# Normalize (scale pixel values 0-255 to 0-1)
# img_array = img_array / 255.0

# Add batch dimension and channel dimension if missing
img_array = np.expand_dims(img_array, axis=0)  # shape (1, 48, 48, 1)

# Predict
pred_probs = model.predict(img_array)
pred_class = np.argmax(pred_probs, axis=1)[0]

print(f"Predicted class index: {classes[pred_class]}")

class_prob_dict = {cls: prob for cls, prob in zip(classes, pred_probs[0])}
for cls, prob in sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{cls}: {prob:.4f}")
