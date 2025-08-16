import os
import pickle

# Path to your dataset folder (update this!)
dataset_path = r"C:\Users\prati\OneDrive\project4\data\train" 

# Get class labels from folder names (sorted for consistency)
class_labels = sorted(os.listdir(dataset_path))

# Save to pickle file
with open("class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)

print("✅ Saved class labels:", class_labels)