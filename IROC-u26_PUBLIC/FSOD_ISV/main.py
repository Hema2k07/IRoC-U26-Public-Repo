# =========================================================
# Few-Shot Learning with Image Similarity & Verification
# ASCEND Project - FSOD (Software Subsystem)
#
# Purpose:
# - Learn from only 5 rock images (few-shot learning)
# - Compare new images using similarity score
# - Output MATCH / NO MATCH
# =========================================================

# ----------- Import Required Libraries -----------

import torch                           # Deep learning framework
import torchvision.models as models    # Pretrained CNN models
import torchvision.transforms as transforms  # Image preprocessing
import cv2                             # Image reading and processing
import numpy as np                     # Numerical operations
from sklearn.metrics.pairwise import cosine_similarity  # Similarity calculation
import os                              # File and folder handling

# -----------------------------
# 1. Configuration Section
# -----------------------------

# Folder containing 5 support (training) rock images
SUPPORT_DIR = "support_images"

# Folder containing test (query) images
QUERY_DIR = "query_images"

# Threshold value for verification
# If similarity >= threshold → MATCH
THRESHOLD = 0.75

# -----------------------------
# 2. Image Preprocessing
# -----------------------------

# Define preprocessing steps to make all images uniform
# These steps match the preprocessing used during ResNet training
transform = transforms.Compose([
    transforms.ToPILImage(),              # Convert NumPy image to PIL format
    transforms.Resize((224, 224)),        # Resize image to 224x224
    transforms.ToTensor(),                # Convert image to PyTorch tensor
    transforms.Normalize(                 # Normalize pixel values
        mean=[0.485, 0.456, 0.406],        # ImageNet mean
        std=[0.229, 0.224, 0.225]          # ImageNet standard deviation
    )
])

# Function to load and preprocess an image
def load_image(image_path):
    # Read image from file
    img = cv2.imread(image_path)

    # Check if image exists
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Convert image from BGR (OpenCV) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply preprocessing transformations
    img = transform(img)

    # Add batch dimension (1, C, H, W)
    return img.unsqueeze(0)

# -----------------------------
# 3. Load Pretrained CNN Model
# -----------------------------

print("[INFO] Loading pretrained ResNet18 model...")

# Load ResNet18 model pretrained on ImageNet
model = models.resnet18(pretrained=True)

# Remove the final classification layer
# We only want feature embeddings, not class labels
model.fc = torch.nn.Identity()

# Set model to evaluation mode
model.eval()

# -----------------------------
# 4. Few-Shot Training Phase
# -----------------------------
# In few-shot learning, we DO NOT train from scratch.
# Instead, we extract and store feature embeddings
# from the 5 support images.

print("[INFO] Creating support embeddings...")

support_embeddings = []                 # List to store feature vectors
support_files = os.listdir(SUPPORT_DIR) # Read support image filenames

# Ensure at least 5 support images are available
if len(support_files) < 5:
    raise ValueError("At least 5 support images are required!")

# Disable gradient calculation (faster + memory efficient)
with torch.no_grad():
    for file in support_files:
        # Full path to support image
        path = os.path.join(SUPPORT_DIR, file)

        # Load and preprocess image
        img = load_image(path)

        # Extract feature embedding using CNN
        embedding = model(img)

        # Convert tensor to NumPy array and store
        support_embeddings.append(embedding.numpy())

# Convert list of embeddings into a single NumPy array
support_embeddings = np.vstack(support_embeddings)

print("[INFO] Support embeddings shape:", support_embeddings.shape)

# -----------------------------
# 5. Query Image Verification
# -----------------------------

print("\n[INFO] Starting verification on query images...\n")

# Read query image filenames
query_files = os.listdir(QUERY_DIR)

# Disable gradient calculation for inference
with torch.no_grad():
    for file in query_files:
        # Full path to query image
        query_path = os.path.join(QUERY_DIR, file)

        # Load and preprocess query image
        query_img = load_image(query_path)

        # Extract feature embedding of query image
        query_embedding = model(query_img).numpy()

        # -----------------------------
        # Similarity Calculation
        # -----------------------------

        # Compute cosine similarity between
        # query embedding and all support embeddings
        similarity_scores = cosine_similarity(
            query_embedding,
            support_embeddings
        )

        # Select the highest similarity score
        max_score = np.max(similarity_scores)

        # -----------------------------
        # Verification Decision
        # -----------------------------

        # If similarity score >= threshold → MATCH
        if max_score >= THRESHOLD:
            decision = "MATCH (Same Rock Material)"
        else:
            decision = "NO MATCH (Different Material)"

        # Display results
        print("--------------------------------------------")
        print(f"Query Image   : {file}")
        print(f"Similarity    : {max_score:.3f}")
        print(f"Verification  : {decision}")

print("\n[INFO] Verification completed successfully.")
