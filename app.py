import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sqlite3
import json
import os
from datetime import datetime
from fuzzy import calculate_reliability

MODEL_FILE = "model/plant_model.pth"
CLASS_FILE = "model/class_names.json"
DATABASE_FILE = "plants.db"
UPLOAD_FOLDER = "uploads"
IMAGE_SIZE = 224

st.set_page_config(
    page_title="Medicinal Plant AI",
    page_icon="🌿"
)

st.title("🌿 Medicinal Plant AI")
st.write("Upload a leaf image to identify the medicinal plant.")

device = torch.device("cpu")
st.info("AI model is running on CPU.")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.isfile(MODEL_FILE):
    st.error("Model file not found: " + MODEL_FILE)
    st.stop()

if not os.path.isfile(CLASS_FILE):
    st.error("Class file not found: " + CLASS_FILE)
    st.stop()

try:
    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        class_data = json.load(file)

    if isinstance(class_data, dict):
        class_names = [
            name
            for name, index in sorted(
                class_data.items(),
                key=lambda item: item[1]
            )
        ]
    else:
        class_names = class_data

except Exception as error:
    st.error("Could not read class_names.json")
    st.exception(error)
    st.stop()


def initialize_database():
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            image_path TEXT,
            plant_name TEXT,
            confidence REAL,
            fuzzy_score REAL,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


initialize_database()


@st.cache_resource
def create_model():
    model = models.mobilenet_v3_small(weights=None)

    number_of_classes = len(class_names)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        number_of_classes
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


try:
    model = create_model()

except Exception as error:
    st.error("The AI model could not be loaded.")
    st.exception(error)
    st.stop()


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def get_plant_info(plant_name):
    try:
        connection = sqlite3.connect(DATABASE_FILE)
        cursor = connection.cursor()

        display_name = plant_name.replace("_", " ")

        cursor.execute("""
            SELECT
                common_name,
                scientific_name,
                description,
                traditional_uses,
                precautions
            FROM plants
            WHERE LOWER(common_name) = ?
        """, (display_name.lower(),))

        result = cursor.fetchone()
        connection.close()

        return result

    except Exception:
        return None


def save_prediction(
    image_name,
    image_path,
    plant_name,
    confidence,
    fuzzy_score
):
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO predictions (
            image_name,
            image_path,
            plant_name,
            confidence,
            fuzzy_score,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        image_path,
        plant_name,
        confidence,
        fuzzy_score,
        created_at
    ))

    connection.commit()
    connection.close()


def get_prediction_history():
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            image_name,
            image_path,
            plant_name,
            confidence,
            fuzzy_score,
            created_at
        FROM predictions
        ORDER BY id DESC
    """)

    results = cursor.fetchall()
    connection.close()

    return results


uploaded_file = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Plant Image",
            width=400
        )

        original_name = uploaded_file.name

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        extension = os.path.splitext(
            original_name
        )[1]

        saved_name = timestamp + extension

        image_path = os.path.join(
            UPLOAD_FOLDER,
            saved_name
        )

        with open(image_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = prediction.item()
        confidence_value = confidence.item() * 100

        if predicted_index >= len(class_names):
            st.error(
                "Model prediction does not match class names."
            )
            st.stop()

        plant_name = class_names[predicted_index]

        display_name = plant_name.replace(
            "_",
            " "
        ).title()

        fuzzy_score = calculate_reliability(
            confidence_value
        )

        save_prediction(
            original_name,
            image_path,
            plant_name,
            confidence_value,
            fuzzy_score
        )

        st.success(
            "🌿 Predicted Plant: " + display_name
        )

        st.metric(
            "AI Confidence",
            f"{confidence_value:.2f}%"
        )

        st.metric(
            "Fuzzy Reliability",
            f"{fuzzy_score:.2f}%"
        )

        st.subheader("Prediction Details")

        probabilities_list = probabilities[0].tolist()

        sorted_predictions = sorted(
            enumerate(probabilities_list),
            key=lambda item: item[1],
            reverse=True
        )

        for index, probability in sorted_predictions[:3]:
            if index < len(class_names):
                name = class_names[index].replace(
                    "_",
                    " "
                ).title()

                st.write(
                    f"{name}: {probability * 100:.2f}%"
                )

        info = get_plant_info(plant_name)

        if info:
            st.subheader("🌱 Plant Information")

            st.write(f"**Common Name:** {info[0]}")
            st.write(f"**Scientific Name:** {info[1]}")
            st.write(f"**Description:** {info[2]}")
            st.write(f"**Traditional Uses:** {info[3]}")

            st.warning(
                f"**Precautions:** {info[4]}"
            )
        else:
            st.info(
                "Plant information was not found in the database."
            )

        st.success(
            "✅ Photo and prediction saved successfully."
        )

    except Exception as error:
        st.error(
            "An error occurred while processing the image."
        )
        st.exception(error)


st.divider()

st.header("📜 Prediction History")

history = get_prediction_history()

if len(history) == 0:
    st.info("No prediction history yet.")
else:
    for record in history:
        image_name = record[0]
        image_path = record[1]
        plant_name = record[2]
        confidence = record[3]
        fuzzy_score = record[4]
        created_at = record[5]

        display_name = plant_name.replace(
            "_",
            " "
        ).title()

        with st.expander(
            f"🌿 {display_name} - {created_at}"
        ):
            st.write(f"**Image:** {image_name}")

            if os.path.isfile(image_path):
                st.image(
                    image_path,
                    width=250
                )

            st.write(
                f"**Predicted Plant:** {display_name}"
            )

            st.write(
                f"**AI Confidence:** {confidence:.2f}%"
            )

            st.write(
                f"**Fuzzy Reliability:** {fuzzy_score:.2f}%"
            )

            st.write(
                f"**Date & Time:** {created_at}"
            )
