#Here codes structure the data

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

input_file = os.path.join(RAW_DIR, "tcga_RSEM_gene_tpm.gz")
output_file = os.path.join(PROCESSED_DIR, "processed_tcga.csv")

print("Loading dataset...")

df = pd.read_csv(input_file, sep='\t', compression='gzip')

print("Original shape:", df.shape)

# Remove missing values

df = df.dropna()

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Normalize values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

processed_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

processed_df.to_csv(output_file, index=False)

print("Processed data saved!")
print("Saved to:", output_file)