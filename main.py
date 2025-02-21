import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import kaggle

# Baixar o dataset usando a API do Kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='.', unzip=True)

# Carregar o dataset
df = pd.read_csv('creditcard.csv')

# Análise Exploratória de Dados
sns.countplot(df['Class'])
plt.title('Distribuição das Classes')
plt.show()

# Pré-processamento
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Balanceamento de Classes
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Modelagem
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliação do Modelo
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.show()
print(classification_report(y_test, y_pred))

# Otimização do Modelo
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Salvar o Modelo
joblib.dump(best_model, 'fraud_detection_model.pkl')
