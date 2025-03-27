import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


X = np.random.rand(1000, 2)
y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


epochs = 100


for i in range(epochs):
    print(f"Essai {i+1}/{epochs}...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, verbose=0)
    
    
    train_accuracy = history.history['accuracy'][0]
    val_accuracy = history.history['val_accuracy'][0]
    
    print(f"Train Accuracy: {train_accuracy*100:.2f}% | Validation Accuracy: {val_accuracy*100:.2f}%")

print("Entraînement terminé après 100 essais.")
