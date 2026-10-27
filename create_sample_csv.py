import pickle
import pandas as pd

# Load the selected features
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Create sample network traffic records
sample_data = pd.DataFrame(
    0,
    index=range(5),
    columns=selected_features
)

# Set some example values for common network features
sample_values = {
    "Destination Port": [80, 443, 53, 22, 8080],
    "Flow Duration": [1000, 2000, 500, 3000, 1500],
    "Total Backward Packets": [2, 5, 1, 3, 4],
    "Total Length of Fwd Packets": [500, 1200, 100, 800, 600],
    "Total Length of Bwd Packets": [300, 900, 80, 400, 350],
}

for feature, values in sample_values.items():
    if feature in sample_data.columns:
        sample_data[feature] = values

# Save the sample CSV
sample_data.to_csv("sample_network_traffic.csv", index=False)

print("Sample CSV created successfully!")
print("Number of features:", len(sample_data.columns))
print(sample_data.head())