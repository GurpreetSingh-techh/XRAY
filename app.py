import os

# Reduce TensorFlow noise and memory use (must be BEFORE importing tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import io
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download

# Use fewer CPU threads (small servers have very little CPU)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# DOWNLOAD MODEL FROM HUGGING FACE
# --------------------------------------------------

MODEL_PATH = hf_hub_download(
    repo_id="GurpreetSingh21/x_ray_detection_system_model",
    filename="pneumonia_densenet121.keras"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading Pneumonia Detection Model...")

# The model was saved with a newer Keras. Older Keras does not know the
# "quantization_config" setting, so we make Dense ignore it.
# (This patch works with any Keras version.)
_original_dense_init = tf.keras.layers.Dense.__init__


def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
    _original_dense_init(self, *args, **kwargs)


tf.keras.layers.Dense.__init__ = _patched_dense_init


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# --------------------------------------------------
# BUILD ONE FAST FUNCTION: PREDICTION + GRAD-CAM
# (built only ONCE when the server starts)
# --------------------------------------------------

def build_gradcam_function():

    backbone = None
    last_conv_layer = None

    # Find the DenseNet backbone and its last Conv2D layer
    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            conv_layers = [
                item for item in layer.layers
                if isinstance(item, tf.keras.layers.Conv2D)
            ]

            if conv_layers:

                backbone = layer
                last_conv_layer = conv_layers[-1]
                break

    if backbone is None:

        raise ValueError("No nested convolutional backbone found.")

    # Split the layers into: before backbone / after backbone
    layers_before = []
    layers_after = []
    found_backbone = False

    for layer in model.layers:

        if layer is backbone:

            found_backbone = True
            continue

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if found_backbone:
            layers_after.append(layer)
        else:
            layers_before.append(layer)

    grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)
        ]
    )
    def run(x):

        h = x

        for layer in layers_before:
            h = layer(h, training=False)

        with tf.GradientTape() as tape:

            conv_outputs, h = grad_model(h, training=False)

            for layer in layers_after:
                h = layer(h, training=False)

            if h.shape[-1] == 1:
                class_score = h[:, 0]
            elif h.shape[-1] == 2:
                class_score = h[:, 1]
            else:
                raise ValueError("Unsupported model output shape.")

        gradients = tape.gradient(class_score, conv_outputs)

        return h, conv_outputs, gradients

    return run


gradcam_function = None

try:

    gradcam_function = build_gradcam_function()

    # Warm-up: run once now, so the first user request is not slow
    gradcam_function(tf.zeros((1, 224, 224, 3), dtype=tf.float32))

    print("Grad-CAM function ready!")

except Exception as setup_error:

    print("Grad-CAM setup failed:", repr(setup_error))

    gradcam_function = None

    # Warm up normal prediction instead
    model.predict(
        np.zeros((1, 224, 224, 3), dtype=np.float32),
        verbose=0
    )


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# TURN MODEL OUTPUT INTO A RESULT
# --------------------------------------------------

def build_result(prediction):

    prediction = np.asarray(prediction)

    # Binary sigmoid output
    if prediction.size == 1:

        pneumonia_score = float(prediction.reshape(-1)[0])

    # Two-class softmax output
    elif prediction.ndim == 2 and prediction.shape[-1] == 2:

        pneumonia_score = float(prediction[0][1])

    else:

        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    pneumonia_score = float(np.clip(pneumonia_score, 0, 1))

    normal_score = 1 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

        description = (
            "The model output is more consistent with "
            "the PNEUMONIA class. This is an educational "
            "AI result, not a medical diagnosis."
        )

    else:

        result = "NORMAL"

        description = (
            "The model output is more consistent with "
            "the NORMAL class. This is an educational "
            "AI result, not a medical diagnosis."
        )

    return {
        "prediction": result,
        "normal_score": round(normal_score * 100, 2),
        "pneumonia_score": round(pneumonia_score * 100, 2),
        "description": description
    }


# --------------------------------------------------
# GRAD-CAM HELPERS (no matplotlib needed)
# --------------------------------------------------

def jet_colormap(values):

    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)

    return np.stack([red, green, blue], axis=-1)


def make_gradcam_image(image_array, conv_outputs, gradients):

    conv_outputs = conv_outputs.numpy()[0]

    pooled_gradients = gradients.numpy().mean(axis=(0, 1, 2))

    heatmap = np.maximum(
        (conv_outputs * pooled_gradients).sum(axis=-1),
        0
    )

    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    heat_values = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    coloured_image = Image.fromarray(
        np.uint8(jet_colormap(heat_values) * 255)
    )

    original = Image.fromarray(
        np.clip(image_array[0], 0, 255).astype(np.uint8)
    ).convert("RGB")

    overlay = Image.blend(
        original,
        coloured_image,
        alpha=0.40
    )

    buffer = io.BytesIO()

    overlay.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --------------------------------------------------
# ANALYSE ONE IMAGE
# --------------------------------------------------

def analyse_image(image_array):

    # Best path: one model pass gives prediction AND Grad-CAM
    if gradcam_function is not None:

        try:

            output, conv_outputs, gradients = gradcam_function(
                tf.constant(image_array)
            )

            result = build_result(output.numpy())

            try:

                if gradients is None:
                    raise ValueError("No gradients returned.")

                result["gradcam"] = make_gradcam_image(
                    image_array,
                    conv_outputs,
                    gradients
                )

            except Exception as grad_error:

                print("Grad-CAM Error:", repr(grad_error))

                result["gradcam"] = None
                result["gradcam_message"] = (
                    "Grad-CAM is unavailable for this image."
                )

            return result

        except Exception as fast_error:

            print("Fast path failed:", repr(fast_error))

    # Backup path: prediction only
    prediction = model.predict(image_array, verbose=0)

    result = build_result(prediction)

    result["gradcam"] = None
    result["gradcam_message"] = (
        "Grad-CAM is unavailable for this model architecture."
    )

    return result


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "Please upload a chest X-ray image."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400

        image = Image.open(file.stream)

        if image.format not in ("JPEG", "PNG"):

            return jsonify({
                "error": "Please upload a JPG or PNG image."
            }), 400

        image.load()

        image = image.convert("RGB")

        image_array = preprocess_image(image)

        result = analyse_image(image_array)

        return jsonify(result)

    except UnidentifiedImageError:

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception as error:

        print("Prediction Error:", repr(error))

        return jsonify({
            "error": "The server could not analyse this image. "
                     "Please try again."
        }), 500


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    # Hugging Face Spaces always uses port 7860
    from waitress import serve

    serve(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        threads=2,
        channel_timeout=300
    )