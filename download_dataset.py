import gdown
import os

# Nouvel ID du fichier sur Google Drive
FILE_ID = "1xHbwrJsjWTyQ3DmsRAZkKVSxnUYNb9iA"
OUTPUT_PATH = "creditcard_2023.csv"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Si le fichier n'existe pas déjà, on le télécharge
if not os.path.exists(OUTPUT_PATH):
    print("Téléchargement du dataset en cours...")
    gdown.download(URL, OUTPUT_PATH, quiet=False)
    print("Téléchargement terminé !")
else:
    print("Le fichier est déjà là, pas besoin de le télécharger.")
