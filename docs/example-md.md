# Notes Sur Les Réseaux De Neurones

## Introduction aux Réseaux de Neurones

Les réseaux de neurones sont des modèles mathématiques inspirés du fonctionnement du cerveau humain. Ils sont composés de plusieurs couches de neurones artificiels interconnectés.

### Architecture de Base

- Couche d'entrée : reçoit les données brutes
- Couches cachées : traitent l'information
- Couche de sortie : produit le résultat final

## Fonctions d'Activation

Les fonctions d'activation courantes incluent :

1. ReLU (Rectified Linear Unit)

   - f(x) = max(0, x)
   - Avantage : évite le problème de la disparition du gradient
   - Utilisée dans les couches cachées

2. Sigmoid
   - f(x) = 1 / (1 + e^(-x))
   - Sortie entre 0 et 1
   - Utile pour la classification binaire

## Exemple de Code Python

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Bonnes Pratiques

Pour éviter le surapprentissage :

- Utiliser la régularisation
- Implémenter du dropout
- Augmenter la taille du jeu de données
- Utiliser la validation croisée

## Références et Ressources

Sites recommandés :

- [Fast.ai](https://www.fast.ai)
- [Coursera Deep Learning](https://www.coursera.org/specializations/deep-learning)
- PyTorch Documentation

## Questions Fréquentes

Q: Quelle est la différence entre ML et DL ?
R: Le Deep Learning est une sous-catégorie du Machine Learning qui utilise des réseaux de neurones profonds avec plusieurs couches cachées.

Q: Combien de données sont nécessaires ?
R: Cela dépend de la complexité du problème, mais généralement plusieurs milliers d'exemples sont nécessaires pour de bons résultats.

## Notes Personnelles

À étudier plus tard :

- Transformers
- Architecture LSTM
- Réseaux de neurones convolutifs (CNN)

TODO: Implémenter un exemple de classification d'images

