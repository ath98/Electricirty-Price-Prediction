from sklearn.ensemble import RandomForestRegressor
from test import data
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
import pandas as pd
import statsmodels.api as sm
import time

df = data()
df.drop(['ImbalanceCost', 'ImbalanceVolume', 'Price First Difference',
        'Seasonal First Difference'], axis=1, inplace=True)
train, test = train_test_split(df, test_size=0.3, random_state=42)

x_train = train.drop(['ImbalancePrice'], axis=1)
y_train = train['ImbalancePrice']

x_test = test.drop(['ImbalancePrice'], axis=1)
y_test = test['ImbalancePrice']

scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


def knn():
    # K nearest regression
    mae = []  # to store rmse values for different k

    # Finding the best value for K
    for K in range(20):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)

        model.fit(x_train, y_train)  # fit the model
        pred = model.predict(x_test)  # make prediction on test set
        error = sqrt(mean_absolute_error(y_test, pred))  # calculate rmse
        mae.append(error)  # store rmse values
        # print('RMSE value for k= ' , K , 'is:', error)

    curve = pd.DataFrame(mae)  # elbow curve
    curve.plot()
    print(mae.index(min(mae)))

    # let's use 5 nearest neighbors as it has the lowest RMSE value
    model = neighbors.KNeighborsRegressor(
        n_neighbors=mae.index(min(mae))+1)

    model.fit(x_train, y_train)  # fit the model
    zeros = [0 for i in range(0,len(y_train))]
    pred = model.predict(x_test)  # make prediction on test set
    org = y_train.tolist()+y_test.tolist()
    predNew = zeros+pred.tolist()
    plt.figure(figsize=(12,8))
    plt.plot(org, label='Original data')
    plt.plot(predNew, '-', label='Fitted data')
    plt.legend()
    plt.show()
    # Measuring accuracy on Testing Data
    print('Accuracy of KNN Regression Model:', model.score(x_test, y_test))
    print('Mean Absolute Error of KNN:', mean_absolute_error(y_test, pred))


def rf():
    startTime = time.time()
    # Random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,bootstrap=True, criterion='mae',
                                max_features=1, min_samples_leaf=1,min_samples_split=3)
    # Hyperparameters
    grid = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid={"max_depth": [3, None],
                                                                                      "max_features": [1, 3, 10],
                                                                                      "min_samples_split": [2, 3, 10],
                                                                                      "min_samples_leaf": [1, 3, 10],
                                                                                      "bootstrap": [True, False],
                                                                                      "criterion": ['absolute_error'],
                                                                                      "n_estimators": [100, 300],
                                                                                      "n_jobs": [-1],
                                                                                      "verbose": [0]},
                        cv=TimeSeriesSplit(n_splits=5), n_jobs=-1)
    # print(grid.get_params())
    
    # grid.fit(x_train, y_train)

    # print("Best parametters", grid.best_params_)

    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    endTime = time.time()
    exeTime = endTime - startTime
    print('Execution time of RF:', exeTime)
    print('Mean Absolute Error of RF:', mean_absolute_error(y_test, pred))
    # print('Accuracy of Random Forest Regression Model:', rf.score(x_test, y_test))
    zeros = [0 for i in range(0,len(y_train))]
    org = y_train.tolist()+y_test.tolist()
    predNew = zeros+pred.tolist()
    # plt.figure(figsize=(12,8))
    # plt.plot(org, label='Original data')
    # plt.plot(predNew, '-', label='Fitted data')
    # plt.legend()
    # plt.show()


rf()
knn()
