import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix

# Загрузка данных
df = pd.read_csv('train.csv')

# Заполнение пропущенных значений
# Возраст средним
df['Age'] = df['Age'].fillna(df['Age'].mean())

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

# Разделение данных
y = df['Transported'].astype(int)
X = df.drop('Transported', axis=1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели дерева решений
clf = DecisionTreeClassifier(max_depth=7, min_samples_split=5, min_samples_leaf=4, random_state=42)
clf.fit(X_train, y_train)

# Прогнозирование
y_pred = clf.predict(X_test)

# Оценка precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Decision Tree)')
plt.show()

# Визуализация дерева решений
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Transported', 'Transported'])
plt.title('Decision Tree Visualization')
plt.show()
