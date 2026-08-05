import json
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def export_model_data(model_path, scaler_path, le_path, output_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    tree = model.tree_
    
    # Normalize tree values to get probabilities at leaves
    # tree.value shape: (node_count, 1, num_classes)
    raw_values = tree.value[:, 0, :]
    sums = raw_values.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1
    probs = (raw_values / sums).tolist()

    # We only need to export the probabilities for leaf nodes to save space, 
    # but for simplicity in Monkey C traversal, we can keep them for all or use a null.
    # However, Garmin RAM is tight. Let's optimize.
    
    # f: feature index (-2 or -1 means leaf)
    # t: threshold
    # l: left child index
    # r: right child index
    # p: probability distribution (only for leaves, others can be empty)
    
    features = tree.feature.tolist()
    thresholds = tree.threshold.tolist()
    left_children = tree.children_left.tolist()
    right_children = tree.children_right.tolist()
    
    # To save space, let's only store the probabilities for LEAF nodes
    # and use the index in a separate list.
    leaf_probs = []
    node_to_prob_idx = [-1] * tree.node_count
    
    for i in range(tree.node_count):
        if left_children[i] == -1: # It's a leaf
            node_to_prob_idx[i] = len(leaf_probs)
            leaf_probs.append([round(p, 4) for p in probs[i]])

    ciq_payload = {
        "m": { # model
            "f": features,
            "t": [round(float(x), 4) for x in thresholds],
            "l": left_children,
            "r": right_children,
            "pi": node_to_prob_idx, # probability index
            "p": leaf_probs # probability values
        },
        "s": { # scaler
            "m": [round(float(x), 4) for x in scaler.mean_.tolist()],
            "s": [round(float(x), 4) for x in scaler.scale_.tolist()]
        },
        "l": le.classes_.tolist() # labels
    }

    with open(output_path, 'w') as f:
        json.dump(ciq_payload, f, separators=(',', ':'))
    
    print(f"Exported model data to {output_path}")
    print(f"Nodes: {tree.node_count}, Leaves: {len(leaf_probs)}, Classes: {len(le.classes_)}")

if __name__ == "__main__":
    # These paths are based on typical output from the training scripts
    import os
    model_dir = os.path.join("models", "decision_tree")
    export_model_data(
        os.path.join(model_dir, "dt_model_subset.joblib"), 
        os.path.join(model_dir, "scaler_subset.joblib"), 
        os.path.join(model_dir, "label_encoder_subset.joblib"), 
        "model_data.json"
    )
