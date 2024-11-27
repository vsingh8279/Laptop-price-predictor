#importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor


#importing the cleaned data
data = pd.read_csv('Cleaned_data.csv')

#splitting the data into price(output) and other specification(input)
X = data.drop(columns=['Price'])
y = np.log(data['Price'])

st.title("Laptop Evaluator: A Futuristic Price Predictor 💻")

# brand
company = st.selectbox('Brand',data['Company'].unique())

# # type of laptop
type = st.selectbox('Type',data['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size',15.4)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',data['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',data['Gpu brand'].unique())

os = st.selectbox('OS',data['os'].unique())



if st.button('PRESS ME 💻'):


    #using the Voting regressor to train the model
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')

    #putting bootstrap=true because we want to use max_sample
    rf = RandomForestRegressor(bootstrap=True,n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
    gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
    xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
    et = ExtraTreesRegressor(bootstrap=True,n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

    #giving each algorithm there weightage
    step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

    model = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    #model training on the given data
    model.fit(X,y)


    # converting some of the input into numerical data
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
  
    #formula for conversion of resolution to ppi
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    #preparing query array which contains all specifications user need
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = np.array(query, dtype=object)
    query = query.reshape(1,12)
    st.title("The predicted price of your selected configuration is INR " + str(int(np.exp(model.predict(query)[0]))))