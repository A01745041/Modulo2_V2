# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:21:14 2023

@author: A0174
"""


import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler


# Cargar los datos
data = pd.read_csv("C:/Users/A0174/OneDrive/Documentos/breastcancerwinsconsin.csv")

# Mapear las etiquetas 'B' y 'M' a 0 y 1 respectivamente
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Imputar valores faltantes con la mediana
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo KNN con regularización L2
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')  # Añade 'weights='distance''
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular las métricas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Calcular la especificidad
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print('\nConfusion Matrix:')
print(conf_matrix_df)

# Imprimir las métricas
print(f'\nPrecision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')


# Generar curva de aprendizaje
train_sizes = np.linspace(0.1, 1.0, 10)  # Tamaños de entrenamiento del 10% al 100%
train_sizes, train_scores, test_scores = learning_curve(clf, X, y, train_sizes=train_sizes, cv=5)

# Calcular la media y desviación estándar de los puntajes
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotear la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del Conjunto de Entrenamiento")
plt.ylabel("Puntaje")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="k", label="Puntaje de Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="c", label="Puntaje de Validación")

plt.legend(loc="best")
plt.show()


# Dividir el conjunto de prueba en prueba y validación (50% prueba, 50% validación)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Evaluar el modelo en el conjunto de entrenamiento
train_accuracy = accuracy_score(y_train, clf.predict(X_train))

# Evaluar el modelo en el conjunto de prueba
test_accuracy = accuracy_score(y_test, clf.predict(X_test))

# Evaluar el modelo en el conjunto de validación
val_accuracy = accuracy_score(y_val, clf.predict(X_val))

# Crear la gráfica de pastel
labels = ['Entrenamiento', 'Prueba', 'Validación']
sizes = [len(y_train), len(y_test), len(y_val)]
colors = ['#ff9999','#66b3ff','#99ff99']
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Distribución de Conjuntos de Datos")

# Imprimir las precisiones
print(f'Precisión en el conjunto de entrenamiento: {train_accuracy}')
print(f'Precisión en el conjunto de prueba: {test_accuracy}')
print(f'Precisión en el conjunto de validación: {val_accuracy}')

# Mostrar la gráfica
plt.show()
