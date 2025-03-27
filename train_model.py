import tensorflow as tf
import numpy as np
import pandas as pd
import os
from datetime import datetime


save_dir = r"C:\Users\charl\Desktop\Projet_IA_CYBER"
os.makedirs(save_dir, exist_ok=True)


X = np.random.rand(1000, 2)
y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),  
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  
              loss='binary_crossentropy',
              metrics=['accuracy'])


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  


target_accuracy = 0.9  
max_epochs = 100  

best_accuracy = 0

for epoch in range(1, max_epochs + 1):
    print(f"Essai {epoch}/{max_epochs}...")

   
    history = model.fit(X, y, validation_split=0.2, epochs=1, verbose=1, callbacks=callbacks)

   
    val_accuracy = history.history['val_accuracy'][-1]

   
    print(f"Train Accuracy: {history.history['accuracy'][-1]*100:.2f}% | Validation Accuracy: {val_accuracy*100:.2f}%")

  
    if val_accuracy >= target_accuracy and best_accuracy < val_accuracy:
        print(f"Objectif atteint à l'essai {epoch} avec une accuracy de validation de {val_accuracy*100:.2f}%.")
        best_accuracy = val_accuracy


print(f"Entraînement terminé avec une accuracy de validation de {best_accuracy*100:.2f}%.")
