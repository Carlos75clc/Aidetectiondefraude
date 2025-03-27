import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from fpdf import FPDF
import webbrowser  

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import seaborn as sns

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

# séparation des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(2,), kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compilation du modèle avec un scheduler de learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ajout de callbacks pour l'optimisation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# entraînement du modèle sur 100 essais
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=1, callbacks=[early_stopping, reduce_lr])

# récupération des résultats de l'entraînement pour chaque epoch
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

# Calcul de la courbe ROC et de l'AUC
y_val_pred_prob = model.predict(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
roc_auc = auc(fpr, tpr)

# Visualisation de la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.grid(True)
roc_img_path = os.path.join(save_dir, f"roc_{timestamp}.png")
plt.savefig(roc_img_path, dpi=300)
plt.close()

# génération du doc pdf avec le compte rendu
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, "Compte Rendu de l'Entrainement", ln=True, align='C')
pdf.ln(10)

# ajouter les infos du compte rendu
pdf.set_font("Arial", size=12)
for line in [
    f"Date : {timestamp}",
    f"Dernier Train Loss : {train_loss[-1]:.4f}",
    f"Dernier Val Loss : {val_loss[-1]:.4f}",
    f"Dernier Train Acc : {train_acc[-1]*100:.2f}%",
    f"Dernier Val Acc : {val_acc[-1]*100:.2f}%",
    f"AUC : {roc_auc:.2f}",
    "",
    "Analyse :",
    "- Si la perte diminue, c'est bon signe.",
    "- Si Val Loss est trop haute, risque d'overfitting.",
    "- On peut ajuster l'optimiseur ou régulariser.",
    "- EarlyStopping et ReduceLROnPlateau ajoutés pour améliorer l'accuracy."
]:
    pdf.cell(200, 8, line, ln=True)

# ajouter les images au pdf
pdf.ln(10)
try:
    pdf.image(graph_img_path, x=10, w=180)
    pdf.image(roc_img_path, x=10, w=180)
except RuntimeError as e:
    print(f"Erreur lors de l'ajout de l'image dans le pdf : {e}")

# sauvegarder le pdf
pdf.output(pdf_path)

# ouvrir l'image du graphique après avoir généré le pdf
webbrowser.open(graph_img_path)

# message de fin
average_accuracy = np.mean(val_acc) * 100  # calcul de l'accuracy moyenne
top_accuracy = np.max(val_acc) * 100  # top accuracy

print(f"Entrainement terminé !")
print(f"Compte rendu PDF : {pdf_path}")
print(f"\nRésumé des résultats :")
print(f" - Précision moyenne (Validation) : {average_accuracy:.2f}%")
print(f" - Meilleure précision obtenue (Validation) : {top_accuracy:.2f}%")
print(f" - AUC : {roc_auc:.2f}")
