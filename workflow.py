from prefect import flow, task
import os
from PIL import Image
import zipfile
import shutil
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing import preprocessing

@task
def resizer(input_folder="0012", output_folder="0012_resized"):
    """Redimensionne les images dans le dossier d'entrée et les sauvegarde dans le dossier de sortie."""
    print("Redimensionnement des images en 1280x1280...")
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
    

@task
def preprocessing():
    train_loader, val_loader, test_loader = preprocessing()
    return train_loader, val_loader, test_loader

@task
def update_zip(zip_path, new_files, temp_dir="temp_unzip"):
    """
    zip_path : chemin du fichier zip à modifier
    new_files : liste des chemins des fichiers à ajouter dans le zip
    temp_dir : dossier temporaire utilisé pour la décompression
    """

    # 1. Nettoyage du dossier temporaire s'il existe
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 2. Décompression du zip dans le dossier temporaire
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 3. Ajout des nouveaux fichiers (en écrasant les anciens si noms identiques)
    for file_path in new_files:
        if os.path.isfile(file_path):
            dest_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)

    # 4. Recompression (on écrase l'ancien zip)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_full_path = os.path.join(root, file)
                arcname = os.path.relpath(file_full_path, temp_dir)
                new_zip.write(file_full_path, arcname=arcname)

    # 5. Nettoyage
    shutil.rmtree(temp_dir)
    print(f"[OK] {zip_path} mis à jour avec {len(new_files)} nouveau(x) fichier(s).")
    
@task
def new_images_in_datasets(datasets):
    for dataset in datasets:
        update_zip()
        
@task
def choose_device():
    """Choisit le device pour l'entraînement (GPU ou CPU)."""
    if torch.cuda.is_available():
        print("Utilisation du GPU pour l'entraînement.")
        return torch.device("cuda")
    else:
        print("GPU non disponible, utilisation du CPU.")
        return torch.device("cpu")

@task
def train(inception_model,device,train_loader, val_loader,optimizer, criterion, scheduler,epochs=3):
    print("Entrainement du modele...")
    # Initialisation du scaler pour le mixed precision
    scaler = GradScaler()

    num_epochs = epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        inception_model.train()  # Mode entraînement
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Utilisation du autocast pour le mixed precision
            with autocast(device_type='cuda'):  # Contexte pour la précision mixte
                outputs = inception_model(images)
                loss = criterion(outputs.logits, labels)

            # Scaler pour éviter les erreurs de précision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Mise à jour de l'échelle du scaler

            running_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        scheduler.step(train_accuracy)

        # Évaluation sur le jeu de validation
        inception_model.eval()  # Mode évaluation
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type='cuda'):  # Utilisation de l'autocast pendant l'évaluation aussi
                    outputs = inception_model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}", f"Learning Rate: {optimizer.param_groups[0]['lr']:.10f}")

        
@task
def retrain_model(train_loader, val_loader):
    print("Entrainement du modele...")
    # Code pour l'entraînement du modèle
    inception_model = models.inception_v3(pretrained=True)

    # Modifier la dernière couche entièrement connectée pour correspondre au nombre de classes
    num_classes = 6
    inception_model.fc = nn.Linear(inception_model.fc.in_features, num_classes)

    # Envoyer le modèle sur le GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_model.to(device)

    print(inception_model)

    # Définir la fonction de perte, l'optimiseur et le planificateur de taux d'apprentissage
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inception_model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    state_dict = torch.load('/model/inception_weights_version4.pth', map_location=device)

    # Load the state dictionary into the model
    inception_model.load_state_dict(state_dict, strict=False)
    
    for name, param in inception_model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    
    train(inception_model, device, train_loader, val_loader, optimizer, criterion, scheduler)
    
    for name, param in inception_model.named_parameters():
        if "fc" not in name:
            param.requires_grad = True
    
    train(inception_model, device, train_loader, val_loader, optimizer, criterion, scheduler)
    
    return inception_model
    

@task
def evaluate_model(inception_model,device, test_loader):
    print("Evaluation du modele...")
        
        # Initialize lists to store predictions and true labels for test and validation sets
    y_pred_test = []
    y_true_test = []
    y_pred_val = []
    y_true_val = []

    # Evaluate on the test set
    inception_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = inception_model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_true_test.extend(labels.cpu().numpy())


    # Calculate precision, recall, and F1-score for the test set
    precision_test = precision_score(y_true_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_true_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted')

    print(f"Test Set Metrics:")
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test}")
    print(f"F1-score: {f1_test}")

    
    return  precision_test, recall_test, f1_test


@task
def update_model(new_model,old_model):
    new_precision,new_recall,new_f1 = evaluate_model(new_model)
    old_precision,old_recall,old_f1 = evaluate_model(old_model)
    if new_f1 > old_f1:
        print("Le nouveau modèle est meilleur, mise à jour du modèle...")
        # Code pour mettre à jour le modèle
        # Par exemple, sauvegarder le nouveau modèle
        torch.save(new_model.state_dict(),'model/inception_weights_version4.pth')
    else:
        print("Le nouveau modèle n'est pas meilleur, le modèle existant reste inchangé.")

    


@flow
def pipeline():
    print("Démarrage du pipeline...")
    # Étape 0: Choix du device
    device = choose_device()
    
    # Étape 1: Redimensionnement des images
    resizer()

    # Étape 2: Mise à jour du zip avec les nouvelles images
    zip_path = "datasets/dataset.zip"
    new_files = ["0012_resized/image1.jpg", "0012_resized/image2.jpg"]
    
    update_zip(zip_path, new_files)
    # Étape 3: Prétraitement des données
    train_loader, val_loader, test_loader = preprocessing()
    
    # Étape 4: Entraînement du modèle
    train(zip_path, 
                train_folder="0012_resized/train", 
                val_folder="0012_resized/val", 
                test_folder="0012_resized/test",
                optimizer='adam',
                criterion='cross_entropy',
                epochs=3)
    inception_model = retrain_model()
    
    # Étape 5: Évaluation du modèle et mise à jour si nécessaire
    update_model(inception_model, old_model='model/inception_weights_version4.pth')