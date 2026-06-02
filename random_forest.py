from collections import Counter
from imblearn.over_sampling import SMOTE
from pickle import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

dados = pd.read_csv('diabetes.csv')

dados_atributos = dados.drop(columns=['Outcome'])
dados_classe = dados['Outcome']

resampler = SMOTE()
atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)
print('=== FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ===')
class_count = Counter(classes_b)
class_count
dados.columns

atributos_train, atributos_teste, classe_train,classe_test = train_test_split(atributos_b,classes_b, test_size=0.3)

tree = DecisionTreeClassifier(random_state=42)

n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

rf_grid={
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split':min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier()
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions= rf_grid,
    n_iter=10,
    cv = 3,
    verbose=2,
    n_jobs=-1
)
rf_hyperparameters.fit(dados_atributos, dados_classe)

rf = RandomForestClassifier(**rf_hyperparameters.best_params_)

bank = rf.fit(dados_atributos, dados_classe)

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
score_cross = cross_validate(
    rf,
    dados_atributos,
    dados_classe,
    scoring=scoring,
    cv=10,
    verbose=1,
    n_jobs=-1
)

print('Matriz de scores: ', score_cross)