import os
import pandas as pd
import numpy as np

# File paths
ORIGINAL_DATA_PATH = os.path.join("data", "raw", "Crop_recommendation.csv")
NEW_DATA_PATH = os.path.join("data", "raw", "updated_crop_recommendation.csv")

# Biophysical ranges for the 5 farm-compatible trees
TREES_BOUNDS = {
    'bamboo': {
        'N': (80, 120), 'P': (40, 70), 'K': (40, 80),
        'temperature': (18.0, 35.0), 'humidity': (60.0, 85.0),
        'ph': (5.5, 6.8), 'rainfall': (1300.0, 2500.0)
    },
    'teak': {
        'N': (40, 80), 'P': (40, 70), 'K': (50, 90),
        'temperature': (22.0, 40.0), 'humidity': (50.0, 80.0),
        'ph': (6.5, 7.8), 'rainfall': (1250.0, 2800.0)
    },
    'eucalyptus': {
        'N': (20, 60), 'P': (15, 40), 'K': (20, 50),
        'temperature': (15.0, 42.0), 'humidity': (30.0, 75.0),
        'ph': (5.0, 8.0), 'rainfall': (600.0, 1500.0)
    },
    'subabul': {
        'N': (10, 40), 'P': (15, 40), 'K': (20, 40),
        'temperature': (18.0, 38.0), 'humidity': (40.0, 80.0),
        'ph': (6.0, 8.5), 'rainfall': (500.0, 1500.0)
    },
    'poplar': {
        'N': (50, 90), 'P': (30, 60), 'K': (40, 80),
        'temperature': (10.0, 35.0), 'humidity': (45.0, 75.0),
        'ph': (6.0, 8.0), 'rainfall': (700.0, 1400.0)
    }
}

def create_updated_dataset():
    if not os.path.exists(ORIGINAL_DATA_PATH):
        raise FileNotFoundError(f"Original CSV not found at {ORIGINAL_DATA_PATH}")

    df_original = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"Original Data Loaded: {df_original.shape[0]} rows ({df_original['label'].nunique()} crops)")

    new_rows = []
    np.random.seed(42)

    for tree_name, bounds in TREES_BOUNDS.items():
        data = {}
        for feature, (low, high) in bounds.items():
            mean = (low + high) / 2.0
            std = (high - low) / 6.0
            generated_vals = np.random.normal(mean, std, 100)
            data[feature] = np.clip(generated_vals, low, high)
        
        df_tree = pd.DataFrame(data)
        df_tree['label'] = tree_name
        new_rows.append(df_tree)
        print(f"Added 100 synthetic rows for '{tree_name}'")

    # Combine original data + 5 new vegetation classes
    df_updated = pd.concat([df_original] + new_rows, ignore_index=True)
    
    # Save to NEW CSV file
    df_updated.to_csv(NEW_DATA_PATH, index=False)
    print(f"\nSUCCESS! Created new dataset at: {NEW_DATA_PATH}")
    print(f"New Total: {df_updated.shape[0]} rows across {df_updated['label'].nunique()} classes.")

if __name__ == "__main__":
    create_updated_dataset()