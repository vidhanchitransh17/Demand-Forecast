import streamlit as st
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
import datetime
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)
import time
import streamlit_gchart as gchart
import pdb

st.set_page_config(layout="wide")

def color_min(value):
    color = 'red' if value <  0 else  'green'
    return 'color: %s' %color
def color_min(value):
    color = 'red' if value <  0 else  'green'
    return 'color: %s' %color


st.image('Logo.jpg',width=70)
st.header("Demand  & Sales forecasting")

A1, A2 = st.columns(2)

with A1:
    container = st.container(border=True)
    container.subheader("Please follow the Steps Given below :")
    container.write("1. Enter the number of days you want forecasted data.")
    container.write('2. Click submit button.')
    container.write('3. Output result will be displayed.')



 
predict_df =  pd.read_csv('train.csv')
 
store_id = list(set(predict_df.store.values))
# store_id = ['ST00'+str(x) for x in store_id]
sku_id = list(set(predict_df.item.values))
# sku_id = ['PDID0'+str(x) for x in sku_id]
sku_id = sku_id+['All']

 
Stock = 100
store_id_option = container.selectbox("Select store ID",store_id)
sku_id_option = container.selectbox("Select the SKU",sku_id)
input_data = container.text_input("Enter the number of days")
stock_data = container.text_input("Enter the current stock")


with A2:
    container2 = st.container(border=True)
    predict_df1 = pd.read_csv(r"C:\Users\vchitransh002\Desktop\Codes\Projects\Demand Forecast\train.csv")
    df = predict_df1
    df['date'] = pd.to_datetime(df['date'])

    container2.subheader('Past Sales')

    # Filter options
    ss = store_id_option
    selected_year = container2.selectbox('Select Year', df['date'].dt.year.unique())

    # Filter data
    filtered_df = df[(df['date'].dt.year == selected_year) & 
                    (df['store'] == store_id_option) & 
                    (df['item'] == sku_id_option)]

    container2.line_chart(filtered_df,y = "sales",x = "date")


if (container.button('Submit')):
    container.write('You selected: '+str(store_id_option)+" store and "+str(sku_id_option)+" product ID")
    placeholder = container.image('https://i.gifer.com/ZKZg.gif',width=60)
    time.sleep(3)
    placeholder.empty()
    model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )

    store_id_option = int(store_id_option)
    if sku_id_option == "All":
        predict_df = predict_df[predict_df['store']==store_id_option]
        predict_df = predict_df.groupby('date')['sales'].sum()
        predict_df = predict_df.to_frame()
        predict_df['date'] = predict_df.index
    
    else:
        sku_id_option = int(sku_id_option)
        predict_df = predict_df[(predict_df['store']==store_id_option) & (predict_df['item']==sku_id_option)]
    predict_df=predict_df.rename(columns={"sales":'y'})
    
    predict_df['ds']  =  predict_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    new_df_train = predict_df[:predict_df.shape[0]-90]
    new_df_test = predict_df[-90:]



    new_df_train.shape[0]+new_df_test.shape[0]
    model.fit(new_df_train)
    future_pd = model.make_future_dataframe(
    periods=int(input_data),
    freq='d',
    include_history=False
    )
    next_days = pd.date_range(start=datetime.date.today(), periods=int(input_data))
    # predict over the dataset
    forecast_pd = model.predict(future_pd)
    actual_stock = int(stock_data)
    y_pred = forecast_pd['yhat'].values
    normal_df = pd.DataFrame({"Demand":[],"Current_Stock":[]})
    req_index = -1
    stock = actual_stock
    for i,j in enumerate(y_pred):
        normal_df.loc[i]=[int(j),stock-int(j)]
        stock =stock -int(j)
        if stock<0 and req_index ==-1:
            req_index =i
    normal_df.index = next_days
    df = normal_df.style.applymap(color_min,subset=['Current_Stock'])

    st.subheader("Demand Forecast vs Inventory")
    T1, T2 = st.columns(2)
    with T1:
        st.dataframe(df)
        html = """
                    <p> The current stock will last for <span style="color:red;">$(name)</span> days</p>
                """

        html = html.replace("$(name)", str(req_index))
        # html = html.replace("$(error)", "Something went wrong!")
        st.markdown(html, unsafe_allow_html=True)
    with T2:
        chart1 = st.container(border=True)
        chart1.line_chart(df,y = "Demand")
        


    optimised_df = pd.DataFrame({"Demand":[],"Current Stock":[]})
    adding_avg, adding_max = 0,0
    if req_index !=-1:
        mean = sum(y_pred[req_index:])//len(y_pred[req_index:])
        mx = max(y_pred[req_index:])
        mn = min(y_pred[req_index:])
        adding_mx = (mean + mx)//2 

    print(adding_max, adding_avg ,sum(y_pred[req_index:]),len(y_pred[req_index:]),req_index)
    stock = actual_stock
    opt = []
    # import pdb; pdb.set_trace()
    for i,j in enumerate(y_pred):
        if i < len(y_pred)//2:
            optimised_df.loc[i]=[int(j),int(stock-int(j))]
            print(int(j),stock-int(j),stock -int(j)+adding_mx)
            stock =stock -int(j)+adding_mx
            opt.append(int(adding_mx))
        else:
            optimised_df.loc[i]=[int(j),int(stock-int(j))]
            print(int(j),stock-int(j),stock -int(j)+mean)
            stock =stock -int(j)+ mean
            opt.append(int(mean))

    optimised_df["Import"] = opt
    optimised_df = optimised_df[["Demand","Import","Current Stock"]]
    st.subheader("Demand Forecast vs Inventory with optimised daily import")  
    G1, G2 = st.columns(2)
    with G1:
        optimised_df.index = next_days
        optimised_df = optimised_df.style.applymap(color_min,subset=['Current Stock'])
        st.dataframe(optimised_df)
        html = """
                    <p> The optimal import for first half of duration is <span style="color:blue;">$(maxname)</span></p>
                    <p> The optimal import for second half of duration is <span style="color:blue;">$(name)</span></p>
                """

        html = html.replace("$(maxname)", str(adding_mx))
        html = html.replace("$(name)", str(mean))
        # html = html.replace("$(error)", "Something went wrong!")
        st.markdown(html, unsafe_allow_html=True)
    with G2:
        chart2 = st.container(border=True)
        chart2.line_chart(optimised_df)
    