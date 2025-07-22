import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler


class AugmentedAnimalDataset(Dataset):
    def __init__(self, root_dir, label, transform=None, num_aug=3):
        """
        root_dir : dossier contenant les images d'un animal.
        label : label unique attribué à toutes les images de ce dataset.
        transform : transformation de base (Resize, ToTensor, Normalisation...).
        num_aug : nombre d'augmentations par image.
        """
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.num_aug = num_aug  # Nombre d'images augmentées par image originale

        # Liste des images du dossier
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)
                            if img.endswith(('.png', '.jpg', '.jpeg'))]

        # Définir les augmentations
        self.augmenters = iaa.Sequential([
            iaa.SomeOf((1, 2), [  # Applique 1 à 2 transformations aléatoires
                iaa.Multiply((0.8, 1.2)),  # Éclairage
                iaa.LinearContrast((0.8, 1.2)),  # Contraste
                iaa.MotionBlur(k=(3, 7)),  # Flou mouvement
                iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Bruit
                iaa.AddToHueAndSaturation((-10, 10)),  # Teinte/Couleur
                iaa.Grayscale(alpha=(0.0, 0.5)),  # Noir & blanc
                iaa.Affine(scale=(0.7, 1.3)),  # Zoom
                iaa.Affine(rotate=(-25, 25)),  # Rotation
                iaa.CoarseDropout((0.02, 0.1), size_percent=(0.02, 0.1))  # Masquage aléatoire
            ])
        ])

    def __len__(self):
        return len(self.image_paths) * (self.num_aug + 1)  # Originale + num_aug augmentées

    def __getitem__(self, idx):
        """
        Retourne soit l'image originale, soit une version augmentée.
        """
        # Trouver l'image originale
        orig_idx = idx // (self.num_aug + 1)  # Trouver l'image d'origine
        img_path = self.image_paths[orig_idx]
        image = Image.open(img_path).convert("RGB")

        # Si idx % (num_aug + 1) == 0 → Image originale, sinon → Augmentée
        if idx % (self.num_aug + 1) != 0:
            image = self.augmenters(image=np.array(image))  # Appliquer augmentation
            image = Image.fromarray(image)

        # Appliquer transform (Resize, Normalisation…)
        if self.transform:
            image = self.transform(image)

        return image, self.label
    
    
"""On applique plusieurs transformations sur les images d'entrée. Le redimensionnement dépend du modèle que l'on entraîne/utilise.
  Par exemple ResNet50 prend en entrée des images en 224x224, alors que Inception-v3 prend en entrée des images en 299x299. E
  Ensuite on transforme les images en tensor qu'on normalise pour garder des valeurs sur les trois canaux entre 0 et 1, facilitant le processus."""
transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),

transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

"""Ces différentes classes ont été crées du fait de la différence de provenance de tous ces datasets trouvés sur internet.
  Chaque de dossier de chaque animal est différent, et il a été jugé plus efficace d'adapter rapidement chaque animal à une classe
  pour les télécharger et leur appliquer un pré-traitement. Dans le cas où un plus grand nombre de classe était présent, standardiser
  les dossiers auraient été cette fois-ci la solution la plus pertinente."""

class BadgerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "badger")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 0  # Toutes les images du dossier "badger" auront le label 0

        if self.transform:
            image = self.transform(image)

        return image, label


class DeerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "animal deer")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 1  # Toutes les images du dossier "deer" auront le label 1

        if self.transform:
            image = self.transform(image)

        return image, label


class Fox(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "red_fox")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 2  # Toutes les images du dossier "fox" auront le label 2

        if self.transform:
            image = self.transform(image)

        return image, label


class HedgehogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "hedgehog")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 3 # Toutes les images du dossier "hedgehog" auront le label 3

        if self.transform:
            image = self.transform(image)

        return image, label


class OtterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "otter")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 4  # Toutes les images du dossier "otter" auront le label 4

        if self.transform:
            image = self.transform(image)

        return image, label



# class RabbitDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []

#         # Parcourir le dossier "badger" et collecter les chemins des images
#         class_dir = os.path.join(root_dir, "wood_rabbit")
#         for image_name in os.listdir(class_dir):
#             if image_name.endswith(('.png', '.jpg', '.jpeg')):
#                 self.image_paths.append(os.path.join(class_dir, image_name))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         label = ??  # Toutes les images du dossier "rabbit" auront le label ??

#         if self.transform:
#             image = self.transform(image)

#         return image, label



# class RatDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []

#         # Parcourir le dossier "badger" et collecter les chemins des images
#         class_dir = os.path.join(root_dir, "animal rat")
#         for image_name in os.listdir(class_dir):
#             if image_name.endswith(('.png', '.jpg', '.jpeg')):
#                 self.image_paths.append(os.path.join(class_dir, image_name))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         label = ?? # Toutes les images du dossier "deer" auront le label ??

#         if self.transform:
#             image = self.transform(image)

#         return image, label




class MartenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "images")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 5  # Toutes les images du dossier "marten" auront le label 5

        if self.transform:
            image = self.transform(image)

        return image, label


class WeaselDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Parcourir le dossier "badger" et collecter les chemins des images
        class_dir = os.path.join(root_dir, "images")
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(class_dir, image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 5  # Toutes les images du dossier "marten" auront le label 6

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Fonction pour afficher les dimensions des données dans un DataLoader
def inspect_loader(loader):
    for images, labels in loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of images : {len(loader)}")
        break  # On affiche seulement les dimensions du premier batch

    

def preprocessing():
    
    badger_train_dataset = AugmentedAnimalDataset(root_dir='badger/badger/badger/data/train/badger', label=0, transform=transform_classification,num_aug=1)
    badger_test_dataset = BadgerDataset(root_dir='badger/badger/badger/data/test', transform=transform_classification)
    badger_val_dataset = BadgerDataset(root_dir='badger/badger/badger/data/val', transform=transform_classification)

    deer_train_dataset = AugmentedAnimalDataset(root_dir='deer/deer/deer/data/train/animal deer', label=1, transform=transform_classification,num_aug=3)
    deer_test_dataset = DeerDataset(root_dir='deer/deer/deer/data/test', transform=transform_classification)
    deer_val_dataset = DeerDataset(root_dir='deer/deer/deer/data/val', transform=transform_classification)

    fox_train_dataset = AugmentedAnimalDataset(root_dir='fox/fox/fox/data/train/red_fox', label=2, transform=transform_classification,num_aug=1)
    fox_test_dataset = Fox(root_dir='fox/fox/fox/data/test', transform=transform_classification)
    fox_val_dataset = Fox(root_dir='fox/fox/fox/data/val', transform=transform_classification)

    hedgehog_train_dataset = AugmentedAnimalDataset(root_dir='hedgehog/hedgehog/hedgehog/data/train/hedgehog', label=3, transform=transform_classification,num_aug=2)
    hedgehog_test_dataset = HedgehogDataset(root_dir='hedgehog/hedgehog/hedgehog/data/test', transform=transform_classification)
    hedgehog_val_dataset = HedgehogDataset(root_dir='hedgehog/hedgehog/hedgehog/data/val', transform=transform_classification)

    otter_train_dataset = AugmentedAnimalDataset(root_dir='otter/otter/data/train/otter', label=4, transform=transform_classification,num_aug=1)
    otter_test_dataset = OtterDataset(root_dir='otter/otter/data/test', transform=transform_classification)
    otter_val_dataset = OtterDataset(root_dir='otter/otter/data/val', transform=transform_classification)

    # rabbit_train_dataset = AugmentedAnimalDataset(root_dir='rabbit/rabbit/data/train/wood_rabbit', label=5, transform=transform_classification,num_aug=2)
    # rabbit_test_dataset = RabbitDataset(root_dir='rabbit/rabbit/data/test', transform=transform_classification)
    # rabbit_val_dataset = RabbitDataset(root_dir='rabbit/rabbit/data/val', transform=transform_classification)

    # rat_train_dataset = AugmentedAnimalDataset(root_dir='rat/rat/data/train/animal rat', label=6, transform=transform_classification,num_aug=4)
    # rat_test_dataset = RatDataset(root_dir='rat/rat/data/test', transform=transform_classification)
    # rat_val_dataset = RatDataset(root_dir='rat/rat/data/val', transform=transform_classification)

    # marten_train_dataset = AugmentedAnimalDataset(root_dir='marten/marten/train/images', label=5, transform=transform_classification,num_aug=8)
    # marten_test_dataset = MartenDataset(root_dir='marten/marten/test', transform=transform_classification)
    # marten_val_dataset = MartenDataset(root_dir='marten/marten/valid', transform=transform_classification)

    # weasel_train_dataset = AugmentedAnimalDataset(root_dir='weasel/weasel/train/images', label=5, transform=transform_classification,num_aug=3)
    # weasel_test_dataset = WeaselDataset(root_dir='weasel/weasel/test', transform=transform_classification)
    # weasel_val_dataset = WeaselDataset(root_dir='weasel/weasel/valid', transform=transform_classification)
    
    # Concatenation des datasets de chaque classe
    full_train_dataset = ConcatDataset([badger_train_dataset, deer_train_dataset,fox_train_dataset, hedgehog_train_dataset, otter_train_dataset, weasel_train_dataset])
    full_test_dataset = ConcatDataset([badger_test_dataset, deer_test_dataset,fox_test_dataset, hedgehog_test_dataset, otter_test_dataset,weasel_test_dataset])
    full_val_dataset = ConcatDataset([badger_val_dataset, deer_val_dataset,fox_val_dataset, hedgehog_val_dataset, otter_val_dataset, weasel_val_dataset])

    # Enfin, on "shuffle" ces datasets on les passe en DataLoader pour les faire passer efficacement dans nos modèles.
    train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(full_val_dataset, batch_size=256, shuffle=False)

    class_counts = {}
    for images, labels in train_loader:
        # Flatten labels if it has more than 1 dimension
        labels = labels.view(-1) # this will flatten to 1D
        for label in labels:
            label = label.item() # Assuming labels are tensors
            class_counts[label] = class_counts.get(label, 0) + 1

    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} occurrences")
        

    # Nombre d’occurrences de chaque classe
    class_counts = torch.tensor(list(class_counts.values()), dtype=torch.float)

    # Inverser la fréquence pour donner plus de poids aux classes rares
    class_weights = 1.0 / class_counts

    # Vérifier si la taille de class_weights correspond au nombre de classes
    num_classes = len(class_counts)  # Get the actual number of classes
    print(f"Number of classes detected: {num_classes}")

    # Si class_weights n'a pas assez de poids pour toutes les classes, on redimensionne :
    if len(class_weights) < 9:
        new_weights = torch.ones(9 - len(class_weights), dtype=torch.float) * class_weights.mean()
        class_weights = torch.cat([class_weights, new_weights])

    # Associer un poids à chaque échantillon en fonction de sa classe
    sample_weights = [class_weights[label] for _, label in full_train_dataset]

    # sampler est ce que va permettre d'équilibrer les batchs en fonction de la rareté de chaque classe.
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


    train_loader = DataLoader(full_train_dataset, batch_size=64, sampler=sampler)

    return train_loader, test_loader, val_loader
