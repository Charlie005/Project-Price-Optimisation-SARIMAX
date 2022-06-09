import pandas as pd
import numpy as np
from json import load
import warnings
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
except:
    pass

if __name__ == "__main__":
    
#-------------Reading the config file
    with open('C:\\Users\\amalj\\Desktop\\360\\Project\\eclerx\\config1.json', 'r') as config_f:
        config = load(config_f)
    
    df = config['transaction_data']
    df=pd.read_excel(df)
    df.columns
    pfilter = config['pfilter']
    df = df[df['Product_description']==pfilter]
    df=df.reset_index()
    df.info() 
    datecolumn = config['datecolumn']
    customertype = config['customertypecol']
    qtycolumn = config['qtycolumn']
    netrevenuecol = config['netrevenuecol']
    # Convert 'Invoice date' and 'date' as datetime format
    df[datecolumn] = df[datecolumn].apply(pd.to_datetime)
    
    dfoutput = df
    col_filter = config['col_filter']
    dfoutput=dfoutput.filter(col_filter,axis=1)
    dfoutput.info()  
    dfoutput[netrevenuecol] = dfoutput[netrevenuecol].astype('float')
    dfoutput.sort_values(datecolumn)
    
    dfoutput_weekly = dfoutput.groupby(customertype).resample(config['resample'], label='right', closed = 'right', on=datecolumn).sum().reset_index().sort_values(by=datecolumn)
    
    #dfoutput_week_Mon = dfoutput.groupby("Customer_Type").resample('W-Mon', label='right', closed = 'right', on='Invoice_Date').sum().reset_index().sort_values(by='Invoice_Date')
    
    dfoutput_weekly.reset_index(drop=True,inplace=True)
    
    dfoutput_weekly["Avg_Price"]=dfoutput_weekly[netrevenuecol]/dfoutput_weekly[qtycolumn]
    
    dropcols = config['dropcols']
    dfoutput_weekly.drop(dropcols,axis=1,inplace=True)
    
        
    dfoutput_weekly = dfoutput_weekly[dfoutput_weekly[customertype]==config['customerfilter']]
    dfoutput_weekly.drop(customertype,inplace=True,axis=1)
    
    dfoutput_weekly.isna().sum()
    dfoutput_weekly.dropna(inplace=True)
    
    dfoutput_weekly.set_index(datecolumn,inplace=True)
    def adfuller_test(series, name=''):
        res = adfuller(series, autolag= config['autolag'])    
        p_value = round(res[1], 3) 
    
        if p_value <= config['sig']:
            print(f" {name} : P-Value = {p_value} => Stationary. ")
        else:
            print(f" {name} : P-Value = {p_value} => Non-stationary.")
    
    for name, column in dfoutput_weekly.iteritems():
        adfuller_test(column, name=column.name)
    
    # Rescale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(dfoutput_weekly)

    # Convert dfoutput_weekly back to a Pandas DataFrame, for convenience
    dfoutput_weekly = pd.DataFrame(rescaledX, index=dfoutput_weekly.index, columns=dfoutput_weekly.columns)
    
    
    ########################## Given file with date,QtySold and Avg_Price######################
    #read input files paths from config.
    data = config['final_data']  

    #df_weekly1 = pd.read_excel('C:/Users/amalj/Desktop/360/Project/data.xlsx')
    df_weekly1 = pd.read_excel(data)
    df_weekly1.info()
    df_weekly1 = df_weekly1.resample(config['resample'], on=datecolumn).sum().reset_index().sort_values(by=datecolumn)
    df_weekly1[[datecolumn]] = df_weekly1[[datecolumn]].apply(pd.to_datetime)
    df_weekly1.set_index(datecolumn,inplace=True)
    
    for name, column in df_weekly1.iteritems():
        adfuller_test(column, name=column.name)
    
    rescaledX1 = scaler.fit_transform(df_weekly1)
    
    df_weekly1 = pd.DataFrame(rescaledX1, index=df_weekly1.index, columns=df_weekly1.columns)
    
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    decompose_data = seasonal_decompose(df_weekly1.iloc[:,0],period=4, model="additive")
    decompose_data.plot();
    
    
    ###########################################################################################


    def my_train_sarimax(df_var):
        mod = sm.tsa.statespace.SARIMAX(df_weekly1.iloc[:,0],order =(1,0,1),seasonal_order = (1,0,1,4))
        # mod = sm.tsa.statespace.SARIMAX(
        #                                 df_var.iloc[:,0], # Variable to be predicted
        #                                 order= config['order'],
        #                                 freq= config['freq'] ,
        #                                 exog = df_var.iloc[:,1:] # exogenous variables
        #                                 ,seasonal_order=config['seasonal_order'],
        #                                 enforce_stationarity=config['enforce_stationarity'],
        #                                 enforce_invertibility=config['enforce_invertibility']
        #                                )

        results = mod.fit()
        
        if config['print_summary']:
            print(results.summary().tables[1])
        
        return results

    # Function that compares SARIMAX predictions vs Real values
    # SARIMAX documentation: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    def compare_pred_vs_real(results, df, predict_from, exog_validation = ''):
        # the end parameter can be used to specify the limit of the prediction
        pred = results.get_prediction(start=pd.to_datetime(predict_from), end=pd.to_datetime(config['predict_till_date']), dynamic=False, exog=exog_validation)
        pred_ci = pred.conf_int()

        ################### PLOTTING DATA
        ax = df['2017':].iloc[:,0].plot(label='Observed')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
        
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)

        ax.set_xlabel('Date')
        ax.set_ylabel('qty prod sold')
        plt.legend()

        plt.show()
        
        ####################################
        y_forecasted = pred.predicted_mean
        y_truth = df[predict_from:].iloc[:,0]

        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

        print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
        
        return y_truth, y_forecasted

        
    # Function for parameter tuning
    def find_optimal_params(df_var, i_freq='M', verbose=False):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        min_aic = 99999999999.0
        opt_params = {'order':'', 'seasonal_order':''}
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(
                                                    df_var.iloc[:,0], # Variable to be predicted
                                                    order= param,
                                                    freq= config['freq'] ,
                                                    exog = df_var.iloc[:,1:] # exogenous variables
                                                    ,seasonal_order=param_seasonal,
                                                    enforce_stationarity=config['enforce_stationarity'],
                                                    enforce_invertibility=config['enforce_invertibility']
                                                    )
                    results = mod.fit()
                    if verbose:
                        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                        
                    if results.aic < min_aic:
                        min_aic = results.aic
                        opt_params['order'] = param
                        opt_params['seasonal_order'] = param_seasonal
                except:
                    print("error while fitting model")
                    continue
        print('Min_AIC')
        print(opt_params)


    find_optimal_params(df_weekly1)
    result = my_train_sarimax(df_weekly1)

    ytruth,ypred = compare_pred_vs_real(result, df_weekly1, config['predict_from_date'], exog_validation=df_weekly1['2017-01-23':].iloc[:,1:])
    #ypred.astype(np.int64) - ytruth.astype(np.int64)
    print(ypred - ytruth)

    p = df_weekly1['Avg_Price'].tolist()
    ypreddf = pd.DataFrame({'Pred':ypred,'P':p})
    ypreddf = scaler.inverse_transform(ypreddf)
    df_weekly1real = scaler.inverse_transform(df_weekly1)




