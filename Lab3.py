import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Lab3")

style = ttk.Style()
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('TLabel', font=('Arial', 12))


def read_data_frame():
    path_to_df = os.path.join('DataSet', 'day.csv')
    return pd.read_csv(path_to_df)


df = read_data_frame()
Data = df[["season", "yr",
           "mnth", "holiday", "weekday", "workingday",
           "weathersit", "temp", "atemp", "hum", "windspeed",
           ]]
Data = Data.astype(float)

KeyAnsw = df["cnt"]
KeyAnsw.value_counts()
print(KeyAnsw.value_counts())
trainData_scaled = MinMaxScaler().fit_transform(Data)
trainX, testX, trainY, testY = train_test_split(trainData_scaled, KeyAnsw, random_state=1)

R2Label = ttk.Label(root, text="R2 score: ")
MSELabel = ttk.Label(root, text="MSE score: ")
MAELabel = ttk.Label(root, text="MAE score: ")


def update_label_text(label, text):
    label.config(text=text)


def linear_regression(trainX, testX, trainY, testY):
    linearRegression = LinearRegression()
    classifier_LinearRegression = linearRegression.fit(trainX, trainY)
    predict_LinearRegression = linearRegression.predict(testX)
    r2 = "R2 score: " + str(r2_score(testY, predict_LinearRegression))
    mse = "MSE score: " + str(mean_squared_error(testY, predict_LinearRegression))
    mae = "MAE score: " + str(mean_absolute_error(testY, predict_LinearRegression))
    update_label_text(R2Label, r2)
    update_label_text(MSELabel, mse)
    update_label_text(MAELabel, mae)


def polynomial_regression(trainX, testX, trainY, testY):
    polynomialRegression = PolynomialFeatures(degree=1)
    polyTrain = polynomialRegression.fit_transform(trainX)
    polyTest = polynomialRegression.fit_transform(testX)

    polyRegression = LinearRegression()
    polyRegression.fit(polyTrain, trainY)
    predict_polyRegression = polyRegression.predict(polyTest)
    r2 = "R2 score: " + str(r2_score(testY, predict_polyRegression))
    mse = "MSE score: " + str(mean_squared_error(testY, predict_polyRegression))
    mae = "MAE score: " + str(mean_absolute_error(testY, predict_polyRegression))
    update_label_text(R2Label, r2)
    update_label_text(MSELabel, mse)
    update_label_text(MAELabel, mae)


def random_forest_regression(trainX, testX, trainY, testY):
    randomForestRegression = RandomForestRegressor(n_estimators=127)
    randomForestRegression.fit(trainX, trainY)
    predict_RandomForestRegression = randomForestRegression.predict(testX)
    r2 = "R2 score: " + str(r2_score(testY, predict_RandomForestRegression))
    mse = "MSE score: " + str(mean_squared_error(testY, predict_RandomForestRegression))
    mae = "MAE score: " + str(mean_absolute_error(testY, predict_RandomForestRegression))
    update_label_text(R2Label, r2)
    update_label_text(MSELabel, mse)
    update_label_text(MAELabel, mae)

    plot_graph(randomForestRegression)


def gradient_boost_regression(trainX, testX, trainY, testY):
    gradientBoostingRegression = GradientBoostingRegressor(random_state=0)
    gradientBoostingRegression.fit(trainX, trainY)
    predict_GradientBoostingRegression = gradientBoostingRegression.predict(testX)
    r2 = "R2 score: " + str(r2_score(testY, predict_GradientBoostingRegression))
    mse = "MSE score: " + str(mean_squared_error(testY, predict_GradientBoostingRegression))
    mae = "MAE score: " + str(mean_absolute_error(testY, predict_GradientBoostingRegression))
    update_label_text(R2Label, r2)
    update_label_text(MSELabel, mse)
    update_label_text(MAELabel, mae)


def plot_graph(RandomForestRegression):
    for widget in frame_graph.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

    pd.DataFrame(RandomForestRegression.feature_importances_, index=Data.columns,
                 columns=["Важность признака"]).sort_values("Важность признака").plot.bar(ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


ButtonLinear = ttk.Button(root, text="линейный регрессор", command=lambda: linear_regression(trainX, testX, trainY, testY))
ButtonPolynomial = ttk.Button(root, text="полиномиальный регрессор",
                     command=lambda: polynomial_regression(trainX, testX, trainY, testY))
ButtonRandomForest = ttk.Button(root, text="регрессор Random Forest",
                     command=lambda: random_forest_regression(trainX, testX, trainY, testY))
ButtonGradientBoost = ttk.Button(root, text="GradientBoosting регрессор",
                     command=lambda: gradient_boost_regression(trainX, testX, trainY, testY))

ButtonLinear.grid(row=0, column=0, padx=5, pady=5)
ButtonPolynomial.grid(row=1, column=0, padx=5, pady=5)
ButtonRandomForest.grid(row=2, column=0, padx=5, pady=5)
ButtonGradientBoost.grid(row=3, column=0, padx=5, pady=5)

R2Label.grid(row=0, column=1, padx=5, pady=5)
MSELabel.grid(row=1, column=1, padx=5, pady=5)
MAELabel.grid(row=2, column=1, padx=5, pady=5)

frame_graph = ttk.Frame(root)
frame_graph.grid(row=0, column=2, rowspan=4, padx=20, pady=20) 


root.mainloop()


def print_df_info(data):
    print(data)
    print(data.describe())
    print(data.info())


if __name__ == '__main__':
    print("end.")
