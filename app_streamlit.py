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
from yolov5.utils.general import non_max_suppression
import sys 
# from yolov5.models.common import DetectMultiBackend
os.system("git lfs install && git lfs pull")



# import gdown 

# --- Configuration de la page ---
st.set_page_config(page_title="Classification Animale", page_icon="🐾", layout="centered")

# --- En-tête ---
st.write(f"Python version: {sys.version}")
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
weights_path = os.path.join("model","MegaDetectorV5.pt")
detection_model = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)['model'].float().fuse().eval()
# detection_model = DetectMultiBackend(weights_path, device='cpu')

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
    # img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # Redimensionner et normaliser (taille classique pour YOLOv5 = 640)
    img_resized = cv2.resize(np.array(image), (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # (C,H,W), [0,1]
    img_tensor = img_tensor.unsqueeze(0)  # Ajouter dimension batch
    
    with torch.no_grad():
       results = detection_model(img_tensor)[0]
       
    # Appliquer NMS
    detections = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)[0]

    # Dessiner les détections sur l’image redimensionnée (640x640)
    for *xyxy, conf, cls in detections:
        label = f"{detection_model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img_resized, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(img_resized, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    

    # --- Prédiction ---
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        proba = torch.nn.functional.softmax(outputs[0], dim=0)
        top1 = torch.argmax(proba).item()
        
    # --- Résultats ---
    with col2:
        st.success(f"### 🧠 Classe prédite : `{classes[top1]}`")
        st.markdown("#### 🔍 Probabilités par classe :")
        for i, p in enumerate(proba):
            st.progress(p.item())
            st.write(f"**{classes[i]}** : {p:.2%}")

    # --- Dessiner bbox sur l'image ---
    # Convert back to RGB PIL
    # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_resized)

    # Afficher image annotée
    st.markdown("<p style='text-align: center; color = blue'> Voilà le résultat de la détection faite sur l'image choisie ! </p>", unsafe_allow_html=True)
    st.image(img_pil, caption="Image avec détection", use_container_width=True)

