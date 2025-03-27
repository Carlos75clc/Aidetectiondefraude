import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from fpdf import FPDF

save_dir = r"C:\Users\charl\Desktop\Projet_IA_CYBER"
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_path = os.path.join(save_dir, f"historique_{timestamp}.xlsx")
pdf_path = os.path.join(save_dir, f"rapport_{timestamp}.pdf")

X = np.random.rand(1000, 2)
y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=30, verbose=1)

epochs = list(range(1, len(history.history['loss']) + 1))
df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss'],
    'Train Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
})
df.to_excel(excel_path, index=False)

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot(epochs, history.history['loss'], label='Train Loss', color='blue')
axs[0].plot(epochs, history.history['val_loss'], label='Validation Loss', color='red')
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_title("Loss Evolution")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, history.history['accuracy'], label='Train Accuracy', color='green')
axs[1].plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='orange')
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Accuracy Evolution")
axs[1].legend()
axs[1].grid(True)

graph_img_path = os.path.join(save_dir, f"plot_{timestamp}.png")
plt.savefig(graph_img_path, dpi=300)
plt.close()

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, "Training Report", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", size=12)
lines = [
    f"Date: {timestamp}",
    f"Last Train Loss: {history.history['loss'][-1]:.4f}",
    f"Last Val Loss: {history.history['val_loss'][-1]:.4f}",
    f"Last Train Acc: {history.history['accuracy'][-1]*100:.2f}%",
    f"Last Val Acc: {history.history['val_accuracy'][-1]*100:.2f}%",
    "",
    "Observations:",
    "- Lower loss indicates better training.",
    "- High validation loss may suggest overfitting.",
    "- Consider tuning optimizer or regularization."
]

for line in lines:
    pdf.cell(200, 8, line, ln=True)

pdf.ln(10)
try:
    pdf.image(graph_img_path, x=10, w=180)
except RuntimeError as e:
    print(f"Error adding image to PDF: {e}")

pdf.output(pdf_path)

print(f"Training completed.\nExcel file: {excel_path}\nPDF report: {pdf_path}")