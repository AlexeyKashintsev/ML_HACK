import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

TRAIN_FILE_NAME = "C:/Users/rizhk/OneDrive/Desktop/Data/Train.xlsx"
TEST_FILES_LOCATION = "C:/Users/rizhk/OneDrive/Desktop/test_input/"
TEST_FILE_TEMPLATE = "Test_input_"
TEST_FILES_COUNT = 4445

train_data = pd.read_excel(TRAIN_FILE_NAME)
columns = train_data.columns[1:]
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.fillna(0, inplace=True)
train_data = train_data.iloc[1:].to_numpy()
for i in range(train_data.shape[0]):
    train_data[i][0] = train_data[i][0][2:]
x_train = []
x_test = []
y = []

for file_number in range(TEST_FILES_COUNT):
    data = pd.read_excel(f"{TEST_FILES_LOCATION}{TEST_FILE_TEMPLATE}{file_number+1}.xlsx")
    data = data.to_numpy()
    for i in range(data.shape[0]):
        for j in range(train_data.shape[0]):
            if data[i][0] == train_data[j][0]:
                if data[i][1] == "Forecast":
                    x_test.append(train_data[j])
                else:
                    x_train.append(train_data[j])
                    y.append(data[i][1])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y = np.array(y)

    x_train = np.delete(x_train, 0, 1)
    x_test = np.delete(x_test, 0, 1)
    plt.figure(figsize=(10, 7))
    plt.plot(x_train.T[0], y)
    plt.title("зависимость признака от целевой переменной")
    plt.xlabel(columns[0])
    plt.ylabel("Предсказываемая переменная")
    plt.show()
    model = LinearRegression()
    optimal = model.fit(x_train, y)
    y_predict = optimal.predict(x_test)
    print(y_predict)

    x_train = []
    x_test = []
