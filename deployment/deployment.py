import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pickle
import numpy as np
import pandas as pd
import sklearn
import datetime


data= pd.DataFrame(columns=['User Rating Count','Price','Developer','Size', 'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date','Description'],index=[0])
data_transformed= pd.DataFrame(columns=['User Rating Count','Price','Developer','Size', 'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date','Description'],index=[0])



#lottie function
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def update_with_new_columns(data,list_all_unique,to_be_hot_encoded):
    # Add New Columns.
    updated_data = data
    updated_data = updated_data.reset_index(drop=True)
    for value in list_all_unique:
        new_column_arr = np.zeros(len(data))
        new_column_df = pd.Series(new_column_arr, name=value)
        updated_data = pd.concat([updated_data, new_column_df], axis=1)

    # Remove Old Columns
    for feature in to_be_hot_encoded:
        updated_data.drop(feature, axis=1, inplace=True)

    # Add Ones in the right columns.
    for feature in to_be_hot_encoded:
        for i in range(0, len(data)):
            cell = data[feature].iloc[i]
            if not pd.isnull(cell):
                valuesList = cell.split(',')
                for value in valuesList:
                    if value in updated_data: # Check if 'value' is seen in training operation.     #else: ignore it.
                        value_i = updated_data.columns.get_loc(value)
                        updated_data.iloc[i, value_i] = 1

    # Update.
    return updated_data

def ouput(y):
    if(y==0):
        ret='High'
    elif (y==1):
        ret='Intermediate'
    else:
        ret='Low'   
    return ret


filename = 'OneHotEncodingList'
list_unquie_train=pickle.load( open(filename, 'rb'))

filename = 'Standardization'
sc=pickle.load( open(filename, 'rb'))

filename = 'Random_Forest_model'
clf =pickle.load(open(filename, 'rb'))

st.set_page_config(page_title="Game Rating Prediction")
st.write("# Game Rating Prediction Deployment")
st.write("---")

st.subheader("Please Enter Your Game Info")
animation=load_lottie("https://assets3.lottiefiles.com/packages/lf20_ckmfiykm.json")

with st.container():
    left_container,right_container=st.columns(2)

    with left_container:
       data['Original Release Date'][0]=st.date_input('Original Release Date',min_value=datetime.date(2010, 1, 1))
       data['Current Version Release Date'][0]=st.date_input('Current Version Release Date',min_value=datetime.date(2010, 1, 1))
       data['Size']=st.number_input('Size', 0, 10000000000)
       data['User Rating Count']=st.number_input('user Rating Count', 0, 1000000)
       data['Price']=st.number_input('Price', 0, 1000)
       data['Description'][0]=st.text_input('Description')
       data['Primary Genre'][0]=st.text_input('Primary Genre')
       data['Genres'][0]=st.text_input('Genres')
       data['Developer'][0]=st.text_input('Developer')
    with right_container:
        st_lottie(animation,speed=1,height=600,width=400)
  
    if st.button("Predict"):
       data_transformed=data
       data_transformed['Original Release Date'] = pd.to_datetime(data['Original Release Date'],dayfirst=True).astype('datetime64[ns]').astype('int64')
       data_transformed['Current Version Release Date'] = pd.to_datetime(data['Current Version Release Date'],dayfirst=True).astype('datetime64[ns]').astype('int64')
       data_transformed['Description'] = data_transformed['Description'].replace(len(data_transformed['Description']))
       data_transformed['Description_length'] = data_transformed['Description'].apply(lambda x: len(str(x)))
       data_transformed['Description_word_length'] = [1 if i<= 500 else 2 if 500<i<=1000 else 3 for i in data_transformed['Description_length']]
       data_transformed.drop(['Description', 'Description_length'], axis=1, inplace = True)
       data_transformed['Size'] = data_transformed['Size']/(1024*1024)
       data_transformed['Size'] = [1 if i<= 1 else 2 if 1<i<=10 else 3 for i in data_transformed['Size']]
       #print(data_transformed.isnull().sum().sort_values(ascending=False))
       data_transformed2=update_with_new_columns(data_transformed,list_unquie_train,['Developer','Genres','Primary Genre'])
       data_transformed_final = sc.transform(data_transformed2)
       y_pred = clf.predict(data_transformed_final)
       prediction=ouput(y_pred)
       st.success('Your game rating is {}'.format(prediction))
