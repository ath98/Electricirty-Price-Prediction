import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import time

def interpolation(data):
    # Checking null data
    # print(data.isnull().sum(axis=0))

    # Filling up null values
    interpolated = data.interpolate(method='ffill')

    # Checking null data after interpolation
    # print(interpolated.isnull().sum(axis=0))

    # print('Non-zero values in each column:\n', data.astype(bool).sum(axis=0), sep='\n')
    # print(data.isnull().sum().sum())

    return interpolated

# mean absolute error
def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def data():
    df=pd.read_csv('Balancing and Imbalance Market Cost View.csv')
    df = interpolation(df)
    # Convert Month into Datetime
    df['StartTime']=pd.to_datetime(df['StartTime']) 
    df['EndTime']=pd.to_datetime(df['EndTime']) 
    df.set_index('StartTime',inplace=True)

    # Data visualize
    df.describe()
    test_result=adfuller(df['ImbalancePrice'],maxlag=48)

    # Defferecing lags
    df['Price First Difference'] = df['ImbalancePrice'] - df['ImbalancePrice'].shift(1)
    df['ImbalancePrice'].shift(1)

    df['Seasonal First Difference']=df['ImbalancePrice']-df['ImbalancePrice'].shift(48)
    return df

def hypertune(df):
    maeLst = []
    # List [0,0,0] to [3,3,3]
    variation = [[p,q,r] for p in range(0,4) for q in range(0,4) for r in range(0,4)]
    print(variation)
    for var in variation:
        p,q,r = var[0],var[1],var[2]
        model = ARIMA(df['ImbalancePrice'], order=(p,q,r))
        model_fit = model.fit()
        df['forecast']=model_fit.predict(start='2022-02-28 19:00:00',end='2022-04-11 10:30:00')
        selectedData = df['ImbalancePrice'].tail(4294)
        maeLst.append(mae(df['forecast'],selectedData))
    curve = pd.DataFrame(maeLst)  # elbow curve
    curve.plot()
    plt.show()
    print('Lowest MAE:',min(maeLst),maeLst.index(min(maeLst)))
        

def main():
    startTime = time.time()
    df = data()
    df['Seasonal First Difference'].plot()

    autocorrelation_plot(df['ImbalancePrice'])
    # plt.show()  

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df['Seasonal First Difference'].iloc[48:],lags=48,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_acf(df['Seasonal First Difference'].iloc[48:],lags=48,ax=ax2)
    
    # hypertune(df)
    
    model=ARIMA(df['ImbalancePrice'],order=(1,1,0),enforce_stationarity=False,enforce_invertibility=False)
    model_fit=model.fit()

    model_fit.summary()
    df['forecast']=model_fit.predict(start='2022-02-28 19:00:00',end='2022-04-11 10:30:00')
    endTime = time.time()
    executionTime = endTime - startTime
    selectedData = df['ImbalancePrice'].tail(4294)
    print("MAE of ARIMA:",mae(df['forecast'],selectedData))
    print("Execution Time of ARIMA:",executionTime)
    print('-----------------')
    df[['ImbalancePrice','forecast']].plot(figsize=(12,8))
    plt.figure(figsize=(12,8))
    plt.plot(df['ImbalancePrice'])
    plt.plot(df['forecast'],'-')
    plt.show()

# main()