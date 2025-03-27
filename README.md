# Projet AI_CreditCardFraudDetection

## Description
Le projet AI_CreditCardFraudDetection a pour objectif de développer un modèle d'IA permettant de détecter les transactions bancaires frauduleuses. Le modèle est entraîné sur un dataset contenant des informations de transactions. Un dashboard est également mis en place pour suivre l'entraînement du modèle, afficher des graphiques d'évaluation des performances, et générer des rapports.

## Auteurs
- Carlosclc
- BastosJT

## Instructions d'utilisation
1. **Téléchargement du dataset :**
   - Le dataset sera automatiquement téléchargé depuis Google Drive en utilisant le script `download_data.py`.
   - Il sera téléchargé dans le même dossier que le script sous le nom `creditcard_2023.csv`.

2. **Entraînement du modèle :**
   - Utilisez le script `train_model.py` pour entraîner le modèle sur le dataset. Le modèle est entraîné à l'aide de TensorFlow et les résultats sont enregistrés dans un dossier `historique`.
   - Le script génère également des graphiques d'entraînement et un fichier PDF avec les résultats.

3. **Lancer le Dashboard :**
   - Le dashboard est une interface graphique qui affiche les performances du modèle, les courbes d’entraînement et de validation, ainsi que des statistiques sur les prédictions.

## Avertissement
Ce projet est à des fins éducatives seulement. L'utilisation de ce modèle doit être faite de manière responsable, en particulier dans un contexte professionnel ou bancaire. L'usage à des fins malveillantes, telles que la manipulation de données sensibles ou frauduleuses, est strictement déconseillé et illégal.
