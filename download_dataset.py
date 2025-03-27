import gdown
import os

# ID du fichier sur Google Drive afin d'obtenir le lien de téléchargement direct
FILE_ID = "1boSxYXXjyrGCNm3OVU1g6ezy8W_qvojV"
# Le fichier sera téléchargé dans le même dossier que le script
OUTPUT_PATH = "creditcard_2023.csv"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Si le fichier n'existe pas déjà, on le télécharge
if not os.path.exists(OUTPUT_PATH):
    print("Téléchargement du dataset en cours...")
    gdown.download(URL, OUTPUT_PATH, quiet=False)
    print("Téléchargement terminé !")
else:
    print("Le fichier est déjà là, pas besoin de le télécharger.")
