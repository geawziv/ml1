import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("train.csv")

print("Данные из датасета:")
print(df)

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())

df.fillna({
    'Age': df['Age'].median(),
    'Cabin': df['Cabin'].mode()[0],
    'HomePlanet': df['HomePlanet'].mode()[0],
    'Destination': df['Destination'].mode()[0]
}, inplace=True)

scaler = MinMaxScaler()
cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df[cols] = scaler.fit_transform(df[cols])

df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

df.to_csv("processed_titanic.csv", index=False)

print("Данные из обработанного датасета:")
print(df)

print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())



