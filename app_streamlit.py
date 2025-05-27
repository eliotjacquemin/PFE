import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import requests
from PytorchWildlife.models import detection as pw_detection
import numpy as np
import cv2

# import gdown 

# --- Configuration de la page ---
st.set_page_config(page_title="Classification Animale", page_icon="🐾", layout="centered")

# --- En-tête ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Classification d'animaux 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Téléverse une image d'animal pour obtenir une prédiction ! </p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color : red'> ATTENTION ! Classes disponibles : blaireau, chevreuil, renard, hérisson, loutre et mustélidé !  </p>", unsafe_allow_html=True)

# --- Fonction pour télécharger depuis Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Chargement du modèle ---
model = models.inception_v3(pretrained=True)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# weights_path = "inception_weights_version2.pth"

weights_path = os.path.join("model", "inception_weights_version2.pth")

# if not os.path.exists(weights_path):
#     file_id = "1kvKxbPthFSGj5fLxPMe0K1H9WWX_w0fT"  # Ton ID Google Drive
#     download_file_from_google_drive(file_id, weights_path)

# if not os.path.exists(weights_path):
#     url = "https://drive.google.com/uc?id=1kvKxbPthFSGj5fLxPMe0K1H9WWX_w0fT"
#     gdown.download(url, weights_path, quiet=False)

state_dict = torch.load(weights_path, map_location=torch.device('cpu'),weights_only=False)
model.load_state_dict(state_dict)
model.eval()

# --- Chargement du model de détection ---
weights_path = "MegaDetectorV5.pt"
model = torch.load("MegaDetectorV5.pt", map_location='cpu')['model'].float().fuse().eval()

# --- Classes ---
classes = ['blaireau', 'chevreuil', 'renard', 'hérisson', 'loutre', 'mustélidé']

# --- Prétraitement ---
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Uploader ---
uploaded_file = st.file_uploader("📤 Choisis une image (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Image chargée", use_container_width=True)
        
    # --- Détection ---
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Redimensionner et normaliser (taille classique pour YOLOv5 = 640)
img_resized = cv2.resize(img_rgb, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # (C,H,W), [0,1]
img_tensor = img_tensor.unsqueeze(0)  # Ajouter dimension batch

    # --- Prédiction ---
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        proba = torch.nn.functional.softmax(outputs[0], dim=0)
        top1 = torch.argmax(proba).item()

    # --- Dessiner bbox sur l'image ---
    if conf > 0:
        # Convert PIL to np.array BGR
        img_cv = np.array(image)[:, :, ::-1].copy()  # RGB->BGR
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Convert back to RGB PIL
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Afficher image annotée
        st.image(img_pil, caption="Image avec détection", use_container_width=True)
    else:
        st.info("Aucune détection trouvée.")
