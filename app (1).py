from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle UNE seule fois
model = tf.keras.models.load_model("model_defect.keras")

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)
    return img

@app.route("/predict-image", methods=["POST"])
def predict():

    # Vérification sécurité
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = preprocess(image)

        pred = model.predict(img)

        return jsonify({"prediction": pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
