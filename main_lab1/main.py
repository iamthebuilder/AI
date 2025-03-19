import pandas as pd

pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv')

print("Первые 5 строк датасета:")
print(df.head())

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())

# возраст средним
df['Age'] = df['Age'].fillna(df['Age'].mean())

# остальные числовые величины медианой
numeric_columns_median = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[numeric_columns_median] = df[numeric_columns_median].fillna(df[numeric_columns_median].median())

# столбцы с категориями модой
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Проверка, что пропущенных значений больше нет
print("\nКоличество пропущенных значений после заполнения:")
print(df.isnull().sum())

# Нормализация числовых данных
scaler = MinMaxScaler()
numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Преобразование категориальных данных в численные
categorical_columns_to_encode = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
df = pd.get_dummies(df, columns=categorical_columns_to_encode, drop_first=True)

print("\nДанные после One-Hot Encoding и нормализации:")
print(df.head())

df.to_csv("processed_titanic.csv", index=False)