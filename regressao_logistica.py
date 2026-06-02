import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dados = pd.read_csv('diabetes.csv')

X = dados.drop(columns=['Outcome'])
y = dados['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_scaled, y_train)

previsoes = modelo.predict(X_test_scaled)

acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia do Modelo: {acuracia:.2%}\n")