from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os
import numpy as np

def get_resnet50_model():
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    # move to FP16 to save memory
    model = model.half()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to standard size
    transforms.ToTensor(),          # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), # normalize colors
    transforms.CenterCrop(224)
])

def get_embedding_from_path(image_path):
    """Return embedding from an image/pdf on disk."""
    try:
        model = get_resnet50_model() 
        if image_path.lower().endswith(".pdf"):
            pages = convert_from_path(image_path, dpi=200, poppler_path=r"c:\poppler-25.07.0\Library\bin")
            image = pages[0]
        else:
            image = Image.open(image_path).convert("RGB")

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor)
            features_np = features.squeeze().numpy()
        del model, features, img_tensor
        import gc
        gc.collect()
        return features_np
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_embeddings(samples_folder):
    custom_embeddings = []
    custom_labels = []

    for filename in os.listdir(samples_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".pdf")):
            filepath = os.path.join(samples_folder, filename)
            emb = get_embedding_from_path(filepath)
            if emb is not None:
                custom_embeddings.append(emb)
                custom_labels.append(filename)

    if len(custom_embeddings) > 0:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        np.save(os.path.join(models_dir, "custom_embeddings.npy"), np.array(custom_embeddings))
        np.save(os.path.join(models_dir, "custom_labels.npy"), np.array(custom_labels))
        print(f"Built and saved {len(custom_embeddings)} sample embeddings to disk.")
    else:
        print("No sample files found to build embeddings.")
        
def get_embedding(file_bytes: bytes, filename: str):
    """
    Returns the embedding vector of an uploaded file.
    file: bytes of the uploaded form image
    filename: the original file name, to detect PDF vs image
    """
    try:
        model = get_resnet50_model()
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor)
            features_np = features.squeeze().numpy()

        del model, features, img_tensor
        import gc
        gc.collect()

        return features_np
    except Exception as e:
        print(f'Error processing {filename}: {e}')

def load_custom_embeddings():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    embeddings_file = os.path.join(models_dir, "custom_embeddings.npy")
    labels_file = os.path.join(models_dir, "custom_labels.npy")

    if os.path.exists(embeddings_file) and os.path.exists(labels_file):
        print("Loaded cached embeddings from disk")
        custom_embeddings = np.load(embeddings_file, allow_pickle=True)
        custom_labels = np.load(labels_file, allow_pickle=True)
    else:
        raise FileNotFoundError("Cached embeddings not found. Upload .npy files to the models/ folder in your repo.")

    return custom_embeddings, custom_labels

