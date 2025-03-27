import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from fpdf import FPDF
import webbrowser  # pour ouvrir l'image après

# on crée un dossier pour stocker les résultats
save_dir = os.path.join(os.getcwd(), 'historique')  # dossier 'historique' pour stocker les résultats
os.makedirs(save_dir, exist_ok=True)

# on crée un nom de fichier unique basé sur l'heure actuelle
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pdf_filename = f"historique_{timestamp}.pdf"
pdf_path = os.path.join(save_dir, pdf_filename)

# on crée des données aléatoires pour l'entraînement
X = np.random.rand(1000, 2)
y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)

# création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# entraînement du modèle sur 100 essais
history = model.fit(X, y, validation_split=0.2, epochs=100, verbose=1)

# on récupère les résultats de l'entraînement pour chaque epoch
epochs = list(range(1, len(history.history['loss']) + 1))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# création des graphiques
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# graphique pour la perte
axs[0].plot(epochs, train_loss, label='Train Loss', color='blue')
axs[0].plot(epochs, val_loss, label='Validation Loss', color='red')
axs[0].set_xlabel("Épochs")
axs[0].set_ylabel("Perte")
axs[0].set_title("Évolution de la perte")
axs[0].legend()
axs[0].grid(True)

# graphique pour la précision
axs[1].plot(epochs, train_acc, label='Train Accuracy', color='green')
axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[1].set_xlabel("Épochs")
axs[1].set_ylabel("Précision")
axs[1].set_title("Évolution de la précision")
axs[1].legend()
axs[1].grid(True)

# sauvegarde de l'image
graph_img_path = os.path.join(save_dir, f"graphiques_{timestamp}.png")
plt.savefig(graph_img_path, dpi=300)
plt.close()

# génération du PDF avec le compte rendu
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, "Compte Rendu de l'Entrainement", ln=True, align='C')
pdf.ln(10)

# ajouter les informations du compte rendu
pdf.set_font("Arial", size=12)
for line in [
    f"Date : {timestamp}",
    f"Dernier Train Loss : {train_loss[-1]:.4f}",
    f"Dernier Val Loss : {val_loss[-1]:.4f}",
    f"Dernier Train Acc : {train_acc[-1]*100:.2f}%",
    f"Dernier Val Acc : {val_acc[-1]*100:.2f}%",
    "",
    "Analyse :",
    "- Si la perte diminue, c'est bon signe.",
    "- Si Val Loss est trop haute, risque d'overfitting.",
    "- On peut ajuster l'optimiseur ou régulariser."
]:
    pdf.cell(200, 8, line, ln=True)

# ajouter l'image au PDF
pdf.ln(10)
try:
    pdf.image(graph_img_path, x=10, w=180)
except RuntimeError as e:
    print(f"Erreur lors de l'ajout de l'image dans le PDF : {e}")

# sauvegarder le PDF
pdf.output(pdf_path)

# ouvrir l'image du graphique après avoir généré le PDF
webbrowser.open(graph_img_path)  # enlever file://

# message de fin
average_accuracy = np.mean(val_acc) * 100  # calcul de l'accuracy moyenne
top_accuracy = np.max(val_acc) * 100  # top accuracy

print(f"Entraînement terminé !")
print(f"Compte Rendu PDF : {pdf_path}")
print(f"\nRésumé des résultats :")
print(f" - Précision moyenne (Validation) : {average_accuracy:.2f}%")
print(f" - Meilleure précision obtenue (Validation) : {top_accuracy:.2f}%")