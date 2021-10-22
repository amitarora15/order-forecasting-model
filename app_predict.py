
import warnings
warnings.filterwarnings("ignore")

import os
import datetime

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from dateutil.parser import parse
import streamlit as st
import joblib 

from sklearn.model_selection import train_test_split, learning_curve
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Lasso

import tensorflow as tf                                                                   # importing tensorflow library
from tensorflow import keras 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
import statsmodels.api as sm  ## sm.stats.acorr_ljungbox check autocorrection in series.   #sm.tsa.ARMA -- model

#root_path="/content/drive/MyDrive/Technical/Learnings/Data Science/Capstone/Final Capstone Project"
root_path="./"


st.set_page_config(layout="centered", page_title='Order Logistics', page_icon="(:shark)")
st.title("Supply Chain Target Prediction")
st.text("The app predict Target order based on Brazilian logistic dataset")


st.sidebar.header("Navigation")
tabs = ["Dashboard", "Regression", "Univariate Forecasting", "Multivariate Forecasting with Regression"]
selected_tab=st.sidebar.radio('Navigation',tabs)

st.sidebar.header("Contributors")
st.sidebar.markdown("""
    - Amit Arora 
    - Shubham Singh 
    - Sameer Lowlekar 
    - Sheetal Mandar Kulkarni 
    - Siddharth Chaturvedi""")

def load_data():
    demand_df = pd.read_csv(root_path+'/Input/Daily_Demand_Forecasting_Orders.csv', sep=';')
    df=pd.DataFrame(demand_df['Target (Total orders)'], index=demand_df.index)
    return df, demand_df;

def add_day_name(day_num):
    day_names=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
    return day_names[day_num-1]

def plot_graph(dataset):
    
    demand_df=dataset.copy()
    demand_df['Day_Name']=demand_df['Day of the week (Monday to Friday)'].apply(lambda x: add_day_name(x))
    cat_day_of_week = CategoricalDtype(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ordered=True)
    
    demand_df_day_total_sum=demand_df.groupby('Day_Name')['Target (Total orders)'].sum().reset_index()
    demand_df_day_total_sum['Day_Name'] = demand_df_day_total_sum['Day_Name'].astype(cat_day_of_week)
    demand_df_day_total_sum=demand_df_day_total_sum.sort_values(['Day_Name'])
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,6))
    sns.barplot(demand_df_day_total_sum['Day_Name'], demand_df_day_total_sum['Target (Total orders)'], label='Target Orders', ax=ax1)
    ax1.set_xlabel("Weekday")
    ax1.set_ylabel("Count of order")
    ax1.set_title("Target orders on weekdays")
    
    demand_df_week_total_sum=demand_df.groupby('Week of the month (first week, second, third, fourth or fifth week')['Target (Total orders)'].sum().reset_index()
    sns.barplot(demand_df_week_total_sum['Week of the month (first week, second, third, fourth or fifth week'], demand_df_week_total_sum['Target (Total orders)'], ax=ax2)
    ax2.set_xlabel("Week of month")
    ax2.set_ylabel("Count of order")
    ax2.set_title("Target orders on weeks")
    
    st.pyplot(fig)  

def user_input_data(active_tab, df):
    
    if active_tab == 'Univariate Forecasting':
        prediction_days=st.number_input('No of days to predict',min_value=3,max_value=15)
        extra_dates = [df.index[-1] + d for d in range (1,prediction_days+1)]
        forecast_df = pd.DataFrame(index=extra_dates)
        option = st.selectbox('Which forecasting model would you like to be applied?', ('SARIMA', 'HWES'))
        return forecast_df, option
    elif active_tab == 'Multivariate Forecasting with Regression': 
        prediction_days=st.number_input('No of days to predict',min_value=3,max_value=15)
        extra_dates = [df.index[-1] + d for d in range (1,prediction_days+1)]
        forecast_df = pd.DataFrame(index=extra_dates)
        option = st.selectbox('Which forecasting model would you like to be applied?', ('VARMAX','Facebook Prophet'))
        return forecast_df, option
    elif active_tab == 'Regression':
        Weekofthemonth=st.number_input('Week of the month',min_value=1,max_value=5, value=5)
        Dayoftheweek=st.number_input('Day of the week',min_value=1,max_value=6, value=6)
        Nonurgentorder=st.number_input('Non-urgent order', value=192.116)
        Urgentorder=st.number_input('Urgent order', value=121.106)
        OrdertypeA=st.number_input('Order type A', value=107.568)
        OrdertypeB=st.number_input('Order type B', value=121.152)
        OrdertypeC=st.number_input('Order type C', value=103.180)
        Fiscalsectororders=st.number_input('Fiscal sector orders', value=18.678)
        trafficcontrollerorders=st.number_input('Orders from the traffic controller sector', value=27328)
        Bankingorders1=st.number_input('Banking orders (1)', value=108072)
        Bankingorders2=st.number_input('Banking orders (2)', value=56015)
        Bankingorders3=st.number_input('Banking orders (3)', value=10690)
        Actual_value=st.number_input('Actual Value', value=331.900)
        data={'Week of the month':Weekofthemonth,
                'Day of the week':Dayoftheweek,
                'Non-urgent order':Nonurgentorder,
                'Urgent order':Urgentorder,
                'Order type A' :OrdertypeA,
                'Order type B' :OrdertypeB,
                'Order type C' :OrdertypeC,
                'Fiscal sector orders' :Fiscalsectororders,
                'Orders from the traffic controller sector' :trafficcontrollerorders,
                'Banking orders (1)' :Bankingorders1,
                'Banking orders (2)' :Bankingorders2,
                'Banking orders (3)' :Bankingorders3
        
        }
        
        features=pd.DataFrame(data,index=[0])
        return Actual_value, features


def load_model(active_tab, model_option, dataset, print_model_selection=True):
    if active_tab == 'Univariate Forecasting':
        if model_option == 'SARIMA':
            st.info("You selected SARIMA model for prediction")
            return joblib.load(root_path+"/Saved Models/timeseries_uni.h5")
        else:    
            st.info("You selected HWES (Hot Winter Exp Smoothing) model for prediction")
            return joblib.load(root_path+"/Saved Models/timeseries_hotwinter_uni.h5")
    elif active_tab == 'Multivariate Forecasting with Regression':
        if model_option == 'VARMAX':
            df = dataset.copy()
            df.drop(columns=['Week of the month (first week, second, third, fourth or fifth week','Day of the week (Monday to Friday)','Fiscal sector orders','Orders from the traffic controller sector','Order type A','Order type B','Order type C'], inplace=True)
            train_percentage=80
            train_final_index=round(len(df)*(train_percentage/100))
            train_data, test_data = df[0:train_final_index], df[train_final_index:]
            #st.write(train_data)
            model = VARMAX(endog=train_data, enforce_stationarity=False, enforce_invertibility=False, order=(1,0))
            results = model.fit(disp=False)
            if print_model_selection:
                st.info("You selected VARMAX model for prediction.")
                st.text("Removed 'Day of the week (Monday to Friday)','Fiscal sector orders','Orders from the traffic controller sector','Order type A','Order type B','Order type C' independent variables as per grangercausalitytests")
            return results
        else:
            st.info("Model Demo under deployment")


def evaluate_model_univ(model, forecast_df, dataset):
    forecast_df['Target (Total orders)'] = model.predict(start=forecast_df.index[0], end=forecast_df.index[-1])
    st.subheader('Forecasted orders')
    st.table(np.exp(forecast_df))
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(dataset.index, dataset['Target (Total orders)'], label='Dataset')
    ax.plot(forecast_df.index, np.exp(forecast_df), label='Forecast Data')
    ax.legend(loc='best', fontsize='medium')
    st.pyplot(fig) 

def evaluate_model_multi(model, forecast_df, full_dataset, is_not_dashboard=True):
    forecast_df = model.predict(start=forecast_df.index[0], end=forecast_df.index[-1])
    if is_not_dashboard:
        st.subheader('Forecasted orders')
    st.table(forecast_df)
    if is_not_dashboard:
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(full_dataset.index, full_dataset['Target (Total orders)'], label='Dataset')
        ax.plot(forecast_df.index, forecast_df['Target (Total orders)'], color='black', label='Forecast Data')
        ax.legend(loc='best', fontsize='xx-large')
        st.pyplot(fig)
    return forecast_df    

def compare_with_regression(full_dataset, forecast_df):
    df_reg=full_dataset.copy()
    cont_col=['Non-urgent order','Urgent order','Banking orders (1)','Banking orders (2)','Banking orders (3)']
    scaler = MinMaxScaler()
    df_reg[cont_col] = scaler.fit_transform(df_reg[cont_col])
    df_final = pd.DataFrame(df_reg[cont_col], index=full_dataset.index)
    train_percentage=80
    train_final_index=round(len(df_final)*(train_percentage/100))
    X_train, X_test = df_final[0:train_final_index], df_final[train_final_index:]
    y_train, y_test = full_dataset[['Target (Total orders)']][0:train_final_index], full_dataset[['Target (Total orders)']][train_final_index:]
    lasso_reg = Lasso(alpha=2.5)
    lasso_reg.fit(X_train, y_train)
    X_forecast=forecast_df[['Non-urgent order','Urgent order','Banking orders (1)','Banking orders (2)','Banking orders (3)']]
    y_forecast=forecast_df[['Target (Total orders)']]
    y_forecast.rename(columns={'Target (Total orders)':'Forecasted Target Order'}, inplace=True)
    scaled_X_forecast=scaler.transform(X_forecast)
    y_pred_forecast = lasso_reg.predict(scaled_X_forecast)
    y_forecast['Reg Predicted Order'] = y_pred_forecast.tolist()
    st.subheader("Multivariate Forecasted orders and Predicted orders uing Regression")
    st.table(y_forecast)
    st.info('RMSE forecast and prediction:' + str(np.sqrt(mean_squared_error(y_forecast['Forecasted Target Order'], y_forecast['Reg Predicted Order']))))
    fig, ax = plt.subplots(figsize=(15,6))
    y_forecast.plot(ax=ax)
    st.pyplot(fig)

def evaluate_model_reg(input_data, full_dataset, Actual_value):
    df_ml=full_dataset.copy()
    cont_col=['Non-urgent order','Urgent order','Order type A','Order type B','Order type C','Fiscal sector orders','Orders from the traffic controller sector','Banking orders (1)','Banking orders (2)','Banking orders (3)']
    scaler = MinMaxScaler()
    
    df_ml[cont_col] = scaler.fit_transform(df_ml[cont_col])
    input_data_reg=input_data.copy()
    input_data_reg[cont_col]=scaler.transform(input_data_reg[cont_col])
    linear_reg=joblib.load(root_path + "/Saved Models/linear_reg.h5")
    linear_value=linear_reg.predict(input_data_reg[cont_col])[0][0]
    lasso_reg=joblib.load(root_path + "/Saved Models/lasso_reg.h5")
    lasso_reg_value=lasso_reg.predict(input_data_reg[cont_col])[0]
    ridge_reg=joblib.load(root_path + "/Saved Models/ridge_reg.h5")
    ridge_reg_value=ridge_reg.predict(input_data_reg[cont_col])[0][0]
    st.write('Before svr')
    svr_reg=joblib.load(root_path + "/Saved Models/svr_reg.h5")
    st.write('Before svr load')
    svr_value=svr_reg.predict(input_data_reg[cont_col])[0]
    st.write('Before svr predict')
    svrRbf_reg=joblib.load(root_path + "/Saved Models/svrRbf_reg.h5")
    svrRbf_reg_value=svrRbf_reg.predict(input_data_reg[cont_col])[0]
    dtree_reg=joblib.load(root_path + "/Saved Models/dtree_reg.h5")
    dtree_reg_value=dtree_reg.predict(input_data_reg[cont_col])[0]
    rF_reg=joblib.load(root_path + "/Saved Models/rF_reg.h5")
    rF_reg_value=rF_reg.predict(input_data_reg[cont_col])[0]
    vt_reg=joblib.load(root_path + "/Saved Models/vt_reg.h5")
    vt_reg_value=vt_reg.predict(input_data_reg[cont_col])[0]

    n_train=48
    input_data_dnn=input_data.copy()
    X = full_dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
    trainX, testX = X[:n_train, :], X[n_train:, :]
    scaler = MinMaxScaler()
    trainX= scaler.fit_transform(trainX)
    df_new= scaler.transform(input_data_dnn)
    mlp_model = joblib.load(root_path + "/Saved Models/mlp_regressor_grid_search_best_model.h5")
    y_mlp_predx=mlp_model.predict(df_new)[0]
    keras_loaded_model_keras=keras.models.load_model(root_path + "/Saved Models/mlp_keras_best_model.h5")
    keras_pred = keras_loaded_model_keras.predict(df_new)[0][0]
    #st.write('Actual Value - '+ str(Actual_value))
    data = [
            ['Linear Regression', linear_value, Actual_value-linear_value], 
            ['Lasso Regression', lasso_reg_value, Actual_value-lasso_reg_value], 
            ['Ridge Regression', ridge_reg_value, Actual_value-ridge_reg_value], 
            ['Linear SVR', svr_value, Actual_value-svr_value],
            ['SVR RBF', svrRbf_reg_value, Actual_value-svrRbf_reg_value],
            ['Decision Tree', dtree_reg_value, Actual_value-dtree_reg_value],
            ['Random Forest', rF_reg_value, Actual_value-rF_reg_value],
            ['Voting Regressor (Linear, Lasso, SVR, SVR RBF)', vt_reg_value, Actual_value-vt_reg_value],
            ['MLP', y_mlp_predx, Actual_value-y_mlp_predx],
            ['DNN Keras', keras_pred, Actual_value-keras_pred]

    ]
    #st.write(data)
    st.table(pd.DataFrame(data, columns = ['Model', 'Predicted Value', 'Error']))

    st.subheader("Feature Importance")
    data={'feature_names':df_ml[cont_col].columns,'feature_importance':rF_reg.feature_importances_}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fig, ax = plt.subplots(figsize=(15,6))
    sns.barplot(x=fi_df['feature_names'], y=fi_df['feature_importance'], ax=ax)
    ax.set_xticklabels(fi_df['feature_names'], Rotation=90, )
    st.pyplot(fig) 
    
dataset, full_dataset=load_data();
if selected_tab == "Dashboard":
    prediction_days=7
    st.subheader('Forecasted Orders for next ' + str(prediction_days) + ' days')
    extra_dates = [dataset.index[-1] + d for d in range (1,prediction_days+1)]
    forecast_df = pd.DataFrame(index=extra_dates)
    model=load_model("Multivariate Forecasting with Regression", 'VARMAX', full_dataset, False)
    evaluate_model_multi(model, forecast_df, full_dataset, False)
    st.subheader('EDA Findings')
    plot_graph(full_dataset)
    st.balloons()
elif selected_tab == "Multivariate Forecasting with Regression":
    st.header("Multivariate Forecasting with Regression")
    forecast_df, model_option=user_input_data(selected_tab, dataset)
    model=load_model(selected_tab, model_option, full_dataset)
    forecast_df = evaluate_model_multi(model, forecast_df, full_dataset)
    compare_with_regression(full_dataset, forecast_df)
elif selected_tab == "Univariate Forecasting":
    st.header("Univariate Forecasting")
    forecast_df, model_option=user_input_data(selected_tab, dataset)
    model=load_model(selected_tab, model_option, dataset)
    evaluate_model_univ(model, forecast_df, dataset)
elif selected_tab == "Regression":
    st.header("Regression")
    Actual_value, input_data=user_input_data(selected_tab, dataset)
    evaluate_model_reg(input_data, full_dataset, Actual_value)
else:
    st.error("Something has gone terribly wrong.")

