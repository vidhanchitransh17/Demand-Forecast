from flask import Flask, render_template, request, jsonify
import pandas as pd
import datetime
from prophet import Prophet

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Load data
        predict_df = pd.read_csv('train.csv')
        # store_id = list(set(predict_df.store.values))
        # sku_id = list(set(predict_df.item.values))
        # sku_id = sku_id + ['All']

        df = predict_df
        df['date'] = pd.to_datetime(df['date'])
        year = df['date'].dt.year.unique()

        store_ex = pd.read_csv('places.csv')
        store_id = list(store_ex["Places"])

        sku_ex = pd.read_csv('products.csv')
        sku_id = list(sku_ex["Products"])
        sku_id = sku_id + ['All']

        return render_template('index.html', store_id=store_id, sku_id=sku_id,year = year)
    elif request.method == 'POST':
        input_data = int(request.form['input_data'])
        stock_data = int(request.form['stock_data'])
        # year = int(request.form['year'])
        store_id_str = request.form['store_id']
        sku_id_str = request.form['sku_id']

        store_ex = pd.read_csv('places.csv')
        sku_ex = pd.read_csv('products.csv')
        store_id_option = store_ex.loc[store_ex['Places'] == store_id_str, 'Values']
        store_id_option = int(store_id_option.iloc[0])
        sku_id_option = sku_ex.loc[sku_ex['Products'] == sku_id_str, 'Values']
        sku_id_option = int(sku_id_option.iloc[0])



        # Load data
        predict_df = pd.read_csv('train.csv')

        # Load model
        model = Prophet(
            interval_width=0.95,
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        # Preprocess data
        store_id_option = int(store_id_option)
        if sku_id_option == "All":
            predict_df = predict_df[predict_df['store'] == store_id_option]
            predict_df = predict_df.groupby('date')['sales'].sum()
            predict_df = predict_df.to_frame()
            predict_df['date'] = predict_df.index
        else:
            sku_id_option = int(sku_id_option)
            predict_df = predict_df[(predict_df['store'] == store_id_option) & (predict_df['item'] == sku_id_option)]
        predict_df = predict_df.rename(columns={"sales": 'y'})

        predict_df['ds'] = predict_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        new_df_train = predict_df[:predict_df.shape[0] - 90]
        new_df_test = predict_df[-90:]

        new_df_train.shape[0] + new_df_test.shape[0]
        model.fit(new_df_train)
        future_pd = model.make_future_dataframe(
            periods=int(input_data),
            freq='d',
            include_history=False
        )
        forecast_pd = model.predict(future_pd)
        actual_stock = int(stock_data)
        y_pred = forecast_pd['yhat'].values

        next_days = pd.date_range(start=datetime.date.today(), periods=int(input_data))

        normal_df = pd.DataFrame({"Demand": [], "Current_Stock": []})
        req_index = -1
        stock = actual_stock
        for i, j in enumerate(y_pred):
            normal_df.loc[i] = [int(j), stock - int(j)]
            stock = stock - int(j)
            if stock < 0 and req_index == -1:
                req_index = i
        normal_df.index = next_days

        optimised_df = pd.DataFrame({"Demand": [], "Current Stock": []})
        adding_avg, adding_max = 0, 0
        if req_index != -1:
            mean = sum(y_pred[req_index:]) // len(y_pred[req_index:])
            mx = max(y_pred[req_index:])
            mn = min(y_pred[req_index:])
            adding_mx = (mean + mx) // 2
            final = (adding_mx+mean)//2

        stock = actual_stock
        opt = []
        for i, j in enumerate(y_pred):
            if i < len(y_pred) // 2:
                optimised_df.loc[i] = [int(j), int(stock - int(j))]
                stock = stock - int(j) + adding_mx
                opt.append(int(adding_mx))
            else:
                optimised_df.loc[i] = [int(j), int(stock - int(j))]
                stock = stock - int(j) + mean
                opt.append(int(mean))

        optimised_df["Import"] = opt
        optimised_df = optimised_df[["Demand", "Import", "Current Stock"]]
        optimised_df.index = next_days


        # Actual Vs Predicted
        store_id = store_id_option
        sku_id = sku_id_option
        duration = input_data

        dataset = pd.read_csv("train.csv")

        predict_df = dataset[(dataset['store'] == store_id) & (dataset['item'] == sku_id)]

        model = Prophet(
            interval_width=0.95,
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        predict_df = predict_df.rename(columns={'sales': 'y'})
        predict_df['ds'] = predict_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        days = int(duration)
        predict_df_train = predict_df.iloc[:-days]
        predict_df_test = predict_df.iloc[-days:]
        model.fit(predict_df_train)
        future_pd = model.make_future_dataframe(periods=days, freq='d', include_history=False)
        forecast_pd = model.predict(future_pd)
        actuals_pd = predict_df_test['y']
        predicted_pd = forecast_pd['yhat']
        new_df = pd.DataFrame({'Actual_Values': actuals_pd.values, 'Predicted_Values': [int(x) for x in predicted_pd.values], 'Date': forecast_pd.ds})
        actual = new_df.to_dict(orient='index')
        actual_dict = {}
        for key, value in actual.items():
            actual_dict[value['Date']] = {'Actual_Values': value['Actual_Values'], 'Predicted_Values': value['Predicted_Values']}
        predict_dict = {str(key.date()): value for key, value in actual_dict.items()}
        actual_json = {str(date): {"Actual_Values": int(data["Actual_Values"]), "Predicted_Values": int(data["Predicted_Values"])} for date, data in predict_dict.items()}

        # # Year Wise sales
        # df = pd.read_csv("train.csv")
        # df['date'] = pd.to_datetime(df['date'])
        # filtered_df = df[(df['date'].dt.year == year) & (df['store'] == store_id_option) & (df['item'] == sku_id_option)]
        # filter = filtered_df.to_dict(orient='index')
        # filter_dict = {}
        # for key, value in filter.items():
        #     filter_dict[value['date']] = {'sales': value['sales'],'store': value['store'],'item': value['item']}
        # filter_dict = {str(key.date()): value for key, value in filter_dict.items()}
        # filter_json = {str(date): {"sales": int(data["sales"]), "store": int(data["store"]), "item": int(data["item"])} for date, data in filter_dict.items()}
        
        
        # Demand with current stock
        normal = normal_df.to_dict(orient='index')
        normal = {str(key.date()): value for key, value in normal.items()}
        normal_json = {str(date): {"Demand": int(data["Demand"]), "Current_Stock": int(data["Current_Stock"])} for date, data in normal.items()}

        # Optimised
        optimised = optimised_df.to_dict(orient='index')
        optimised = {str(key.date()): value for key, value in optimised.items()}
        for date, data in optimised.items():
            data['optimal'] = data.pop('Current Stock')
        optimised_json = {str(date): {"Demand": int(data["Demand"]), "Import": int(data["Import"]), "optimal": int(data["optimal"])} for date, data in optimised.items()}

        

        # Stock Vs Optimal
        stock_v_optimal = {}

        for date, data in optimised.items():
            stock_v_optimal[date] = {
                'optimal': data['optimal'],
                'Current_Stock': normal[date]['Current_Stock']
            }
        stock_v_optimal_json = {str(date): {"Current_Stock": int(data["Current_Stock"]), "optimal": int(data["optimal"])} for date, data in stock_v_optimal.items()}

        # import pdb;pdb.set_trace()
        last = str(req_index)+ " days."

        return jsonify(stock_v_optimal = stock_v_optimal_json,optimised=optimised_json,actual = actual_json,req_index = last,imp= final,day = str(duration)+" days")

if __name__ == '__main__':
    app.run(debug=True)
