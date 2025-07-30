#################################################################################################################################################
def coimbatore_run() :
    
    import requests, pandas as pd
    import os
    
    # === 0. Initialize CSV structure if file is missing or empty ===
    price_csv_path = r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_tomato_full_dataset.csv"
    if not os.path.exists(price_csv_path) or os.path.getsize(price_csv_path) == 0:
        pd.DataFrame(columns=[
            "Date", "Current_Price", "Market_Name", "Volume_Sold", "Quality_Grade"
        ]).to_csv(price_csv_path, index=False)
        print(" Empty price CSV initialized with header columns.")

    # === 1. Fetch daily weather data ===
    def fetch_weather(location, start, end, key):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}"
        try:
            res = requests.get(url, params={
                "unitGroup": "metric",
                "include": "days",
                "key": key
            })
            res.raise_for_status()
            print(" Weather data fetched successfully.")
            return pd.DataFrame(res.json()["days"])[["datetime", "precip", "tempmax", "humidity"]]
        except Exception as e:
            print(" Weather API failed:", e)
            return pd.DataFrame()

    weather_df = fetch_weather("Coimbatore", "2024-01-01", "2025-07-16", "EYHS73GASM5S3MLQAWTVM8NDB")

    # === 2. Load price data ===
    price_df = pd.read_csv(price_csv_path)
    if not price_df.empty and "Date" in price_df.columns:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df = price_df.set_index("Date").resample("D").ffill().reset_index()
        print(f"️ Price data loaded with {len(price_df)} records.")

        # === 3. Merge and add features ===
        df = weather_df.rename(columns={"datetime": "Date", "precip": "Rain_mm"}).merge(price_df, on="Date")
        df["Season_Avg"] = df["Date"].dt.month.map({
            1: 25, 2: 27, 3: 30, 4: 35,
            5: 38, 6: 36, 7: 32, 8: 31,
            9: 30, 10: 28, 11: 26, 12: 24
        })
        df["Target_Price_T+3"] = df["Current_Price"].shift(-3)
        df = df.dropna(subset=["Target_Price_T+3"])
        print(f" Merged DataFrame has {len(df)} rows.")

        # === 4. Save final dataset ===
        df.to_csv(price_csv_path, index=False)
        print(" Updated dataset saved successfully.")
    else:
        print(" Price data is empty or missing 'Date' column. No processing performed.")
    weather_df.to_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_weather_data.csv", index=False)
    print(" Weather data saved to CSV.")

    print(weather_df.head())
    print(weather_df.shape)

    #################################################################################################################################################

    import pandas as pd
    import numpy as np

    def coimbatur_data():
        # Step 1: Read your real weather CSV
        df_weather = pd.read_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_weather_data.csv")

        # Step 2: Ensure 'datetime' column is parsed as actual dates
        df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

        # Step 3: Generate synthetic tomato prices
        n = len(df_weather)
        base_price = 30 + 2.5 * np.sin(np.linspace(0, 4*np.pi, n))
        noise = np.random.normal(0, 1.2, n)
        spikes = np.zeros(n)
        spikes[[int(n/4), int(n/2), int(3*n/4)]] = [5, -3, 6]  # simulated disruptions

        df_weather['tomato_price'] = np.round(base_price + noise + spikes, 2)

        # Step 4: Forecast target → price after 3 days
        df_weather['price_plus_3'] = df_weather['tomato_price'].shift(-3)

        # Step 5: Drop final rows with NaN target
        df_weather.dropna(inplace=True)
        # Save to file
        df_weather.to_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_weather_price_forecast_ready.csv", index=False)

        # Preview the result
        

    coimbatur_data()

#################################################################################################################################################

    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
 
    # Step 1: Load your weather + tomato price dataset from CSV
    df_weather = pd.read_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_weather_price_forecast_ready.csv")

    # Step 2: Parse datetime column
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    # Step 3: Select input features and target (price after 3 days)
    features = ['precip', 'tempmax', 'humidity']
    X = df_weather[features]
    y = df_weather['price_plus_3']

    # Step 4: Train/test split without shuffling (preserving time order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Step 5: Initialize and train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    import pandas as pd

    # 🔹 Find next date
    last_date = df_weather['datetime'].max()
    next_date = last_date + pd.Timedelta(days=1)

    # 🔹 Estimate weather for the next day
    # (Here we simply take the last recorded values. You could use rolling averages or models instead.)
    latest_weather = df_weather[features].iloc[-1]

    df_input = pd.DataFrame({
        'precip': [latest_weather['precip']],
        'tempmax': [latest_weather['tempmax']],
        'humidity': [latest_weather['humidity']]
    })

    # 🎯 Predict with trained Random Forest model
    predicted_price = model.predict(df_input)[0]

    print(f"\n Date: {next_date.date()}")
    print(f" Predicted coimbatore market tomato price on {next_date.date()}: Rs {predicted_price:.2f} per kg")
    
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def chitore_run() :
    
    import requests, pandas as pd
    import os
    
    # === 0. Initialize CSV structure if file is missing or empty ===
    price_csv_path = r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\chitore_tomato_full_dataset.csv"
    if not os.path.exists(price_csv_path) or os.path.getsize(price_csv_path) == 0:
        pd.DataFrame(columns=[
            "Date", "Current_Price", "Market_Name", "Volume_Sold", "Quality_Grade"
        ]).to_csv(price_csv_path, index=False)
        print(" Empty price CSV initialized with header columns.")

    # === 1. Fetch daily weather data ===
    def fetch_weather(location, start, end, key):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}"
        try:
            res = requests.get(url, params={
                "unitGroup": "metric",
                "include": "days",
                "key": key
            })
            res.raise_for_status()
            print(" Weather data fetched successfully.")
            return pd.DataFrame(res.json()["days"])[["datetime", "precip", "tempmax", "humidity"]]
        except Exception as e:
            print(" Weather API failed:", e)
            return pd.DataFrame()

    weather_df = fetch_weather("Coimbatore", "2024-01-01", "2025-07-16", "EYHS73GASM5S3MLQAWTVM8NDB")

    # === 2. Load price data ===
    price_df = pd.read_csv(price_csv_path)
    if not price_df.empty and "Date" in price_df.columns:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df = price_df.set_index("Date").resample("D").ffill().reset_index()
        print(f"️ Price data loaded with {len(price_df)} records.")

        # === 3. Merge and add features ===
        df = weather_df.rename(columns={"datetime": "Date", "precip": "Rain_mm"}).merge(price_df, on="Date")
        df["Season_Avg"] = df["Date"].dt.month.map({
            1: 25, 2: 27, 3: 30, 4: 35,
            5: 38, 6: 36, 7: 32, 8: 31,
            9: 30, 10: 28, 11: 26, 12: 24
        })
        df["Target_Price_T+3"] = df["Current_Price"].shift(-3)
        df = df.dropna(subset=["Target_Price_T+3"])
        print(f" Merged DataFrame has {len(df)} rows.")

        # === 4. Save final dataset ===
        df.to_csv(price_csv_path, index=False)
        print(" Updated dataset saved successfully.")
    else:
        print(" Price data is empty or missing 'Date' column. No processing performed.")
    weather_df.to_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\chitore_weather_data.csv", index=False)
    print(" Weather data saved to CSV.")

    print(weather_df.head())
    print(weather_df.shape)

#################################################################################################################################################

    import pandas as pd
    import numpy as np

    def coimbatur_data():
        # Step 1: Read your real weather CSV
        df_weather = pd.read_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\chitore_weather_data.csv")

        # Step 2: Ensure 'datetime' column is parsed as actual dates
        df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

        # Step 3: Generate synthetic tomato prices
        n = len(df_weather)
        base_price = 30 + 2.5 * np.sin(np.linspace(0, 4*np.pi, n))
        noise = np.random.normal(0, 1.2, n)
        spikes = np.zeros(n)
        spikes[[int(n/4), int(n/2), int(3*n/4)]] = [5, -3, 6]  # simulated disruptions

        df_weather['tomato_price'] = np.round(base_price + noise + spikes, 2)

        # Step 4: Forecast target → price after 3 days
        df_weather['price_plus_3'] = df_weather['tomato_price'].shift(-3)

        # Step 5: Drop final rows with NaN target
        df_weather.dropna(inplace=True)
        # Save to file
        df_weather.to_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\chitore_weather_price_forecast_ready.csv", index=False)

        # Preview the result
        

    coimbatur_data()
#################################################################################################################################################
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
 
    # Step 1: Load your weather + tomato price dataset from CSV
    df_weather = pd.read_csv(r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\chitore_weather_price_forecast_ready.csv")

    # Step 2: Parse datetime column
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    # Step 3: Select input features and target (price after 3 days)
    features = ['precip', 'tempmax', 'humidity']
    X = df_weather[features]
    y = df_weather['price_plus_3']

    # Step 4: Train/test split without shuffling (preserving time order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Step 5: Initialize and train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    import pandas as pd
    
    # 🔹 Find next date
    last_date = df_weather['datetime'].max()
    next_date = last_date + pd.Timedelta(days=1)

    # 🔹 Estimate weather for the next day
    # (Here we simply take the last recorded values. You could use rolling averages or models instead.)
    latest_weather = df_weather[features].iloc[-1]

    df_input = pd.DataFrame({
        'precip': [latest_weather['precip']],
        'tempmax': [latest_weather['tempmax']],
        'humidity': [latest_weather['humidity']]
    })

    # 🎯 Predict with trained Random Forest model
    predicted_price = model.predict(df_input)[0]

    print(f"\n Date: {next_date.date()}")
    print(f" Predicted chitore market tomato price on {next_date.date()}: Rs {predicted_price:.2f} per kg")

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################





#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################    
print("COIMBATORE : \n")
coimbatore_run()
print("\n")
print("\n")
print("CHITORE : \n")
chitore_run()

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


