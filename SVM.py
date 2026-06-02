import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv('diabetes.csv')

X = dados.drop('Outcome', axis=1)
y = dados['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_svm = SVC(kernel='rbf', C=1.0)
modelo_svm.fit(X_train_scaled, y_train)

y_pred = modelo_svm.predict(X_test_scaled)

print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")