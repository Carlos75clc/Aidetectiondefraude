import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from fpdf import FPDF

# 📂 Créer un dossier pour les résultats
save_dir = r"C:\Users\charl\Desktop\Projet Cyber & IA\Aidetectiondefraude\historique des essais"
os.makedirs(save_dir, exist_ok=True)

# 📂 Nom unique du fichier PDF
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pdf_filename = f"compte_rendu_{timestamp}.pdf"
pdf_path = os.path.join(save_dir, pdf_filename)

# 📉 Génération de données (XOR)
X = np.random.rand(1000, 2)
y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)

# 💡 Définition du modèle avec Dropout et Batch Normalization pour améliorer la précision
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(2,)),
    tf.keras.layers.BatchNormalization(),  # Batch Normalization
    tf.keras.layers.Dropout(0.2),  # Dropout pour éviter l'overfitting
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔄 Entraînement du modèle avec EarlyStopping pour éviter l'overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X, y, validation_split=0.2, epochs=100, verbose=1, callbacks=[early_stopping])

# 📊 Récupération des métriques
epochs = list(range(1, len(history.history['loss']) + 1))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 💨 Création des graphiques
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot(epochs, train_loss, label='Train Loss', color='blue')
axs[0].plot(epochs, val_loss, label='Validation Loss', color='red')
axs[0].set_xlabel("Épochs")
axs[0].set_ylabel("Perte")
axs[0].set_title("Évolution de la perte")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, train_acc, label='Train Accuracy', color='green')
axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[1].set_xlabel("Épochs")
axs[1].set_ylabel("Précision")
axs[1].set_title("Évolution de la précision")
axs[1].legend()
axs[1].grid(True)

# 📝 Sauvegarde des graphiques
graph_img_path = os.path.join(save_dir, f"graphiques_{timestamp}.png")
plt.savefig(graph_img_path, dpi=300)
plt.close()

# 🌟 Génération du PDF avec le compte rendu
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, "Compte Rendu de l'Entrainement", ln=True, align='C')
pdf.ln(10)

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
    "- On peut ajuster l'optimiseur ou régulariser.",
    "- Dropout et Batch Normalization ajoutés pour éviter l'overfitting.",
    "- EarlyStopping activé pour arrêter l'entraînement si la perte ne s'améliore pas."
]:
    pdf.cell(200, 8, line, ln=True)

pdf.ln(10)
try:
    pdf.image(graph_img_path, x=10, w=180)
except RuntimeError as e:
    print(f"Erreur lors de l'ajout de l'image dans le PDF : {e}")

pdf.output(pdf_path)

# 📁 Suppression de l'image temporaire uniquement si le PDF est bien sauvegardé
if os.path.exists(pdf_path):
    os.remove(graph_img_path)

print(f"\u2705 Entraînement terminé !\nCompte Rendu PDF : {pdf_path}")
