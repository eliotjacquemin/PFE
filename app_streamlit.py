import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os 
import urllib.request 

# --- Configuration de la page ---
st.set_page_config(page_title="Classification Animale", page_icon="🐾", layout="centered")

# --- En-tête ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Classification d'animaux 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Téléverse une image d'animal pour obtenir une prédiction !</p>", unsafe_allow_html=True)

# --- Chargement du modèle ---
model = models.inception_v3(pretrained=True)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

weights_path = "inception_weights_version2.pth"

if not os.path.exists(weights_path):
    url = "https://drive.google.com/uc?export=download&id=1kvKxbPthFSGj5fLxPMe0K1H9WWX_w0fT"  # remplace par ton ID
    urllib.request.urlretrieve(url, weights_path)
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

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

