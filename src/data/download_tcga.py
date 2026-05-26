#download the dataset 
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

url = "https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz"

output_file = os.path.join(RAW_DIR, "tcga_RSEM_gene_tpm.gz")

print(f"Downloading file to: {output_file}")

response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download complete!")
else:
    print(f"Failed. Status code: {response.status_code}")