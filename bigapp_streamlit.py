import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
import sys
from yolov5.utils.general import non_max_suppression
from ultralytics import YOLO

# Configuration
os.system("git lfs install && git lfs pull")
st.set_page_config(page_title="Détection & Classification Animale", page_icon="🦊", layout="centered")

# Hooks Grad-CAM
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# En-tête
st.write(f"Python version: {sys.version}")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Détection & Classification d'animaux 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Téléverse une image d'animal pour obtenir une prédiction ! </p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color : red'> ATTENTION ! Classes disponibles : blaireau, chevreuil, renard, hérisson, loutre et mustélidé !  </p>", unsafe_allow_html=True)

# Modèle de classification
model = models.inception_v3(pretrained=True)
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

weights_path = os.path.join("model", "inception_weights_version5.pth")
state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict)
model.eval()

# Enregistrement des hooks Grad-CAM
model.Mixed_7c.register_forward_hook(forward_hook)
model.Mixed_7c.register_backward_hook(backward_hook)

# Modèle de détection
detection_model = YOLO("model/MDV6-yolov9-e-1280.pt")

# Classes
classes = ['blaireau', 'chevreuil', 'renard', 'hérisson', 'loutre', 'mustélidé']

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Upload images
uploaded_files = st.file_uploader("📁 Choisis plusieurs images (JPG ou PNG)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"📷 Image : {uploaded_file.name}"):
            st.markdown(f"---\n### 📷 Image : {uploaded_file.name}")
            col1, col2 = st.columns([1, 2])
            image = Image.open(uploaded_file).convert("RGB")

            with col1:
                st.image(image, caption="Image chargée", use_container_width=True)

            # Détection
            img_resized = cv2.resize(np.array(image), (1280,1280))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                results = detection_model(img_tensor)[0]

            boxes = results.boxes
            gradcam_imgs = []
            
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = f"{results.names[int(cls)]} {conf:.2f}"

                cv2.rectangle(img_resized, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(img_resized, f"{label} #{idx+1}", (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                roi = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                input_tensor = transform(roi).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    proba = torch.nn.functional.softmax(outputs[0], dim=0)
                    top1 = torch.argmax(proba).item()

                with col2:
                    st.success(f"### 🧠 Classe prédite pour la ROI #{idx+1} : {classes[top1]}")
                    st.markdown("#### 🔍 Probabilités par classe :")
                    for i, p in enumerate(proba):
                        st.write(f"**{classes[i]}** : {p:.2%}")
                        st.progress(p.item())

                # Grad-CAM pour chaque ROI
                model.zero_grad()
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                output[0, top1].backward(retain_graph=True)

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * features).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().detach().numpy()
                cam = cv2.resize(cam, (299, 299))
                cam = (cam - cam.min()) / (cam.max() - cam.min())

                # img_cv = cv2.cvtColor(np.array(roi.resize((299, 299))), cv2.COLOR_RGB2BGR)
                # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.4 + img_cv
                # superimposed_img = cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB)
                # gradcam_imgs.append(Image.fromarray(superimposed_img))
                
                # Générer la heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (roi.width, roi.height))

                # Superposition avec transparence (0.4)
                roi_cv = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)
                superimposed = cv2.addWeighted(roi_cv, 0.6, heatmap, 0.2, 0)

                # Intégrer dans l'image de détection (img_resized)
                img_with_cam = img_resized.copy()
                x1, y1, x2, y2 = map(int, xyxy)
                img_with_cam[y1:y2, x1:x2] = superimposed
                img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)


            img_pil = Image.fromarray(img_resized)
            tab1, tab2, tab3 = st.tabs(["Original", "Detected", "Grad-CAM"])

            with tab1:
                st.image(image, use_container_width=True, caption="Image originale")
            with tab2:
                st.image(img_pil, use_container_width=True, caption="Image avec détection")
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().astype(int))
                    width = x2 - x1
                    height = y2 - y1
                    st.write(f"Taille de la bounding box {idx+1}: {width} x {height} pixels")
            with tab3:
                # for idx, gradcam_img in enumerate(gradcam_imgs):
                #     st.image(gradcam_img, use_container_width=True, caption=f"Grad-CAM ROI #{idx+1}")
                st.image(img_with_cam, use_container_width=True, caption="Grad-CAM dans la détection")


st.markdown("""
        <hr>
        <div style='text-align: center;'>
        <small>App développée avec ❤️ par Eliot</small>
        </div>
""", unsafe_allow_html=True)