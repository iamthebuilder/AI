import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('train.csv')

# Заполнение пропущенных значений
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Возраст средним

# Остальные числовые величины медианой
numeric_columns_median = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[numeric_columns_median] = df[numeric_columns_median].fillna(df[numeric_columns_median].median())

# Столбцы с категориями модой
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Удаление ненужных колонок
df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

# Преобразование категориальных данных в численные
categorical_columns_to_encode = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
df = pd.get_dummies(df, columns=categorical_columns_to_encode, drop_first=True)

# Разделение данных для логистической регрессии
X_class = df.drop('Transported', axis=1)
y_class = df['Transported'].astype(int)

# Разделение данных для линейной регрессии
X_reg = df.drop('Age', axis=1)
y_reg = df['Age']

# Разделение на обучающую и тестовую выборки
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Масштабирование данных для логистической регрессии
scaler_class = StandardScaler()
X_train_class = scaler_class.fit_transform(X_train_class)
X_test_class = scaler_class.transform(X_test_class)

# Масштабирование данных для линейной регрессии
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Логистическая регрессия
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_class, y_train_class)

y_pred_class = log_reg.predict(X_test_class)

# Оценка модели логистической регрессии
f1 = f1_score(y_test_class, y_pred_class)
print("Логистическая регрессия (Transported):")
print("F1 Score:", f1)

# Матрица ошибок для логистической регрессии
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# 2. Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

# Предсказание на тестовой выборке
y_pred_reg = lin_reg.predict(X_test_reg)

# Оценка модели линейной регрессии
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("\nЛинейная регрессия (Age):")
print("Среднеквадратичная ошибка (MSE):", mse)

# Визуализация предсказаний
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.xlabel("Фактические значения (Age)")
plt.ylabel("Предсказанные значения (Age)")
plt.title("Фактические vs Предсказанные значения (Age)")
plt.show()