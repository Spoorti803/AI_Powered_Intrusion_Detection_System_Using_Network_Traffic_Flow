import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

files = [
    "xgboost_multiclass.pkl",
    "label_encoder.pkl",
    "selected_features.pkl",
]

for filename in files:
    filepath = os.path.join(BASE_DIR, filename)

    print(f"\nChecking: {filename}")

    if not os.path.exists(filepath):
        print("ERROR: File not found!")
        continue

    try:
        with open(filepath, "rb") as file:
            obj = pickle.load(file)

        print("Loaded successfully!")
        print("Object type:", type(obj).__name__)

        if filename == "xgboost_multiclass.pkl":
            print("Expected feature count:", obj.n_features_in_)

        elif filename == "label_encoder.pkl":
            print("Classes:", obj.classes_)

        elif filename == "selected_features.pkl":
            print("Selected feature count:", len(obj))
            print("First 10 features:", obj[:10])

    except Exception as error:
        print("ERROR:", type(error).__name__, error)