import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#pd.set_option('display.max_columns', None)
df = pd.read_csv("train.csv")

print("Данные из датасета:")
print(df)

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())

df.fillna({
    'HomePlanet': df['HomePlanet'].mode()[0],
    'CryoSleep': df['CryoSleep'].median(),
    'Cabin': df['Cabin'].mode()[0],
    'Destination': df['Destination'].mode()[0],
    'Age': df['Age'].median(),
    'VIP': df['VIP'].mode()[0],
    'RoomService': df['RoomService'].mode()[0],
    'FoodCourt': df['FoodCourt'].mode()[0],
    'ShoppingMall': df['ShoppingMall'].mode()[0],
    'Spa': df['Spa'].mode()[0],
    'VRDeck': df['VRDeck'].mode()[0]
}, inplace=True)

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
df['RoomService'] = scaler.fit_transform(df[['RoomService']])
df['FoodCourt'] = scaler.fit_transform(df[['FoodCourt']])
df['ShoppingMall'] = scaler.fit_transform(df[['ShoppingMall']])
df['Spa'] = scaler.fit_transform(df[['Spa']])
df['VRDeck'] = scaler.fit_transform(df[['VRDeck']])

df = pd.get_dummies(df, columns=['Destination'], drop_first=True)
df = pd.get_dummies(df, columns=['VIP'], drop_first=True)
df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)
df = pd.get_dummies(df, columns=['Transported'], drop_first=True)
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)
df = pd.get_dummies(df, columns=['CryoSleep'], drop_first=True)

df.to_csv("processed_titanic.csv", index=False)

print("Данные из обработанного датасета:")
print(df)

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())




