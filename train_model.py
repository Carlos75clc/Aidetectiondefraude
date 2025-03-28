import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from fpdf import FPDF
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import json

# Charger le dataset
df = pd.read_csv("creditcard_2023.csv")

# Afficher les premières lignes du dataset pour comprendre sa structure
print(df.head())

# Extraire les caractéristiques et le label
X = df.drop('Class', axis=1).values  # Toutes les colonnes sauf 'Class'
y = df['Class'].values  # La colonne 'Class' qui indique si la transaction est frauduleuse (1) ou non (0)

# Normaliser les données (on ne normalise pas la colonne 'Class')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Séparer les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Création du modèle de réseau de neurones
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilation du modèle avec un scheduler de learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Ajout de callbacks pour l'optimisation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Entraînement du modèle
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=1, callbacks=[early_stopping, reduce_lr])

# Récupération des résultats de l'entraînement pour chaque époque
epochs = list(range(1, len(history.history['loss']) + 1))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Création des graphiques
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Graphique pour la perte
axs[0].plot(epochs, train_loss, label='Train Loss', color='blue')
axs[0].plot(epochs, val_loss, label='Validation Loss', color='red')
axs[0].set_xlabel("Épochs")
axs[0].set_ylabel("Perte")
axs[0].set_title("Évolution de la perte")
axs[0].legend()
axs[0].grid(True)

# Graphique pour la précision
axs[1].plot(epochs, train_acc, label='Train Accuracy', color='green')
axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[1].set_xlabel("Épochs")
axs[1].set_ylabel("Précision")
axs[1].set_title("Évolution de la précision")
axs[1].legend()
axs[1].grid(True)

# Sauvegarde de l'image
graph_img_path = os.path.join(os.getcwd(), f"graphiques_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
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
roc_img_path = os.path.join(os.getcwd(), f"roc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.savefig(roc_img_path, dpi=300)
plt.close()

# Génération du doc pdf avec le compte rendu
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, "Compte Rendu de l'Entrainement", ln=True, align='C')
pdf.ln(10)

# Ajouter les infos du compte rendu
pdf.set_font("Arial", size=12)
for line in [
    f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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

# Ajouter les images au PDF
pdf.ln(10)
try:
    pdf.image(graph_img_path, x=10, w=180)
    pdf.image(roc_img_path, x=10, w=180)
except RuntimeError as e:
    print(f"Erreur lors de l'ajout de l'image dans le PDF : {e}")

# Sauvegarder le PDF
pdf_output_path = os.path.join(os.getcwd(), f"historique_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
pdf.output(pdf_output_path)

# Ouvrir l'image du graphique après avoir généré le PDF
webbrowser.open(graph_img_path)
webbrowser.open(pdf_output_path)

# Message de fin
average_accuracy = np.mean(val_acc) * 100  # Calcul de l'accuracy moyenne
top_accuracy = np.max(val_acc) * 100  # Top accuracy

print(f"Entraînement terminé !")
print(f"Compte Rendu PDF : {pdf_output_path}")
print(f"\nRésumé des résultats :")
print(f" - Précision moyenne (Validation) : {average_accuracy:.2f}%")
print(f" - Meilleure précision obtenue (Validation) : {top_accuracy:.2f}%")
print(f" - AUC : {roc_auc:.2f}")

# Sauvegarder les résultats dans un fichier JSON
def save_results(history, auc, file_path='results.json'):
    results = {
        "train_loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "train_acc": history.history['accuracy'],
        "val_acc": history.history['val_accuracy'],
        "auc": auc
    }
    
    with open(file_path, 'w') as f:
        json.dump(results, f)
    
# Après l'entraînement
save_results(history, roc_auc)