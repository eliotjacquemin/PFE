from PIL import Image
import os

# Dossier d'origine et dossier de sortie
input_folder = "0012"
output_folder = "0012_resized"

# Crée le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Redimensionne chaque image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Ouvre, resize, et sauvegarde dans le nouveau dossier
        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((1280, 1280))
        img_resized.save(output_path)

print("✅ Toutes les images ont été redimensionnées et sauvegardées dans :", output_folder)
