"""
Created on Sun Nov 21 15:45:29 2021

UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: INTELIGENCIA ARTIFICIAL
Tema: Proyecto parcial
Alumno: Bryan Michael Trejo Garcia & Renata Monserrat Andrade Ruiz
Profesor: Dr. Asdrúbal López Chau
Descripción: PREDICCIÓN DE ABORTO ESPONTANEO

@author: Bryan Michael
"""


import pandas as pd 
from sklearn.preprocessing import LabelEncoder #Codificar datos de una etiqueta
from sklearn.preprocessing import MinMaxScaler #Rango minimo y maximo
from sklearn.model_selection import train_test_split #Separar datos de entrenamiento y de prueba
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix #Matriz de confusion
from sklearn.metrics import accuracy_score, precision_score #Precison y exactitud que tendra el problema

datos = pd.read_csv('datosAborto.csv') #Datos a usar datosAborto
datos.head()
#Cambiar nombre de los datos columnas
columnas = ['Problemas hormonales', 'Diabetes', 'Consumo de drogas', 'Consumo de alcohol', 'Alimentacion', 'Traumatismo', 'Infeccion', 'E. por T. Sexual', 'Edad', 'Problema de utero']
datos.columns=columnas
datos.head() #Imprimir
#datos.dtypes #De que tipo son los datos que se estan usando
#datos.isnull().sum() #Existencia de datos nulos

ecoder = LabelEncoder() #Transformar las columnas deseadas
datos['Alimentacion'] = ecoder.fit_transform(datos['Alimentacion'])
datos['Problema de utero'] = ecoder.fit_transform(datos['Problema de utero'])
datos.head() #Imprimir
scale = MinMaxScaler(feature_range = (0,100)) #Rango a usar
datos['Problemas hormonales'] = scale.fit_transform(datos['Problemas hormonales'].values.reshape(-1, 1))
datos.head() #Imprimir

y = datos['Problema de utero'] #Variable dependiente
X = datos.drop('Problema de utero', axis= 1) #Variable independiente


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.28, random_state = 1) #Separar prueba y entrenamiento
KNN_model = KNeighborsClassifier(n_neighbors=15) #Numero de puntos a clasificar
KNN_model.fit(X_train, y_train)

y_test_pred = KNN_model.predict(X_test) #Prediccion con el modelo y datos de prueba

print(confusion_matrix(y_test, y_test_pred)) #Obtener matriz de confusion

accuracy_score(y_test, y_test_pred) #Exactitud
print(accuracy_score(y_test, y_test_pred))
precision_score(y_test, y_test_pred) #Precision
print(precision_score(y_test, y_test_pred))
