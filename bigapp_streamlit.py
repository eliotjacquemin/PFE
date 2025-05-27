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

# --- Chargement du modèle ---
model = models.inception_v3(pretrained=True)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# weights_path = "inception_weights_version2.pth"

weights_path = os.path.join("model", "inception_weights_version2.pth")

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

# --- Uploader de dossier ---
uploaded_files = st.file_uploader(
    "📁 Choisis plusieurs images (JPG ou PNG)", 
    type=["jpg", "png"], 
    accept_multiple_files=True
)


if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"📷 Image : {uploaded_file.name}"):
          st.markdown(f"---\n### 📷 Image : `{uploaded_file.name}`")
          col1, col2 = st.columns([1, 2])
          image = Image.open(uploaded_file).convert("RGB")
          with col1:
              st.image(image, caption="Image chargée", use_container_width=True)
          
          # Détection
          img_resized = cv2.resize(np.array(image), (640, 640))
          img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
          img_tensor = img_tensor.unsqueeze(0)
          
          with torch.no_grad():
              results = detection_model(img_tensor)[0]
          
          detections = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)[0]
          
          for idx, (*xyxy, conf, cls) in enumerate(detections):
              label = f"{detection_model.names[int(cls)]} {conf:.2f}"
              cv2.rectangle(img_resized, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
              cv2.putText(img_resized, f"{label} #{idx+1}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
          
          # Classification
          input_tensor = transform(image).unsqueeze(0)
          with torch.no_grad():
              outputs = model(input_tensor)
              proba = torch.nn.functional.softmax(outputs[0], dim=0)
              top1 = torch.argmax(proba).item()
          
          with col2:
              st.success(f"### 🧠 Classe prédite : `{classes[top1]}`")
              st.markdown("#### 🔍 Probabilités par classe :")
              for i, p in enumerate(proba):
                  st.progress(p.item())
                  st.write(f"**{classes[i]}** : {p:.2%}")
          
          # Affichage image détectée
          img_pil = Image.fromarray(img_resized)
          st.image(img_pil, caption="Image avec détection", use_container_width=True)
