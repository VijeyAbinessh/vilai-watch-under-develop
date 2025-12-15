##start##
def coimbatore_run():
    import requests, pandas as pd
    import os
    from datetime import date, timedelta

    # === 1. WeatherAPI (1-day historical fetch) ===
    def fetch_weather_weatherapi(city="Coimbatore", target_date=None):
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")
        url = "http://api.weatherapi.com/v1/history.json"
        params = {
            "key": "bc026ec54af04fa28bc80205252807",  # Your WeatherAPI key
            "q": city,
            "dt": target_date
        }
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            forecast = res.json().get("forecast", {}).get("forecastday", [])
            if forecast:
                data = forecast[0]["day"]
                return pd.DataFrame([{
                    "datetime": target_date,
                    "tempmax": data["maxtemp_c"],
                    "Rain_mm": data["totalprecip_mm"],
                    "humidity": data["avghumidity"]
                }])
            else:
                print(f"⚠️ No forecast data for {target_date}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ WeatherAPI fetch failed for {target_date}:", e)
            return pd.DataFrame()

    # === 2. Wrapper to fetch latest N days ===
    def fetch_weather_ndays_weatherapi(city="Coimbatore", days=1):
        dfs = []
        for i in range(days):
            target_date = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            df = fetch_weather_weatherapi(city=city, target_date=target_date)
            if not df.empty:
                dfs.append(df)
        weather_df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Combined weather data for {days} day(s) — shape: {weather_df.shape}")
        return weather_df

    # === 3. Begin weather fetch and processing ===
    weather_df = fetch_weather_ndays_weatherapi(city="Coimbatore", days=1)

    if not weather_df.empty and "datetime" in weather_df.columns:
        print("📊 Weather Data (latest):")
        print(weather_df[["datetime", "tempmax", "Rain_mm", "humidity"]])
    else:
        print("⚠️ No valid weather data to display.")
        return  # Exit early if no data

    # === 4. Engineer season average and append to CSV ===
    df = weather_df.rename(columns={"datetime": "Date"}).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Season_Avg"] = df["Date"].dt.month.map({
        1: 25, 2: 27, 3: 30, 4: 35,
        5: 38, 6: 36, 7: 32, 8: 31,
        9: 30, 10: 28, 11: 26, 12: 24
    })

    path = r"C:\Users\vijey abinessh\Desktop\shakthi hackathon\coimbatore_weather_data.csv"

    # === 5. Ensure file has valid structure ===
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=["datetime", "tempmax", "Rain_mm", "humidity", "Season_Avg"]).to_csv(path, index=False)
        print("🆕 CSV file created with headers.")

    final_row = df[["Date", "tempmax", "Rain_mm", "humidity", "Season_Avg"]].copy()
    final_row = final_row.rename(columns={"Date": "datetime"})

    try:
        existing = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame(columns=["datetime", "tempmax", "Rain_mm", "humidity", "Season_Avg"])
        print("⚠️ CSV was corrupted or blank. Reinitialized from scratch.")

    if existing.empty:
        updated = final_row.copy()
    else:
        updated = pd.concat([existing, final_row], ignore_index=True)

    updated.to_csv(path, index=False)
    print(f"📥 Appended latest weather row. Total rows now: {len(updated)}")

# === RUN IT ===
coimbatore_run()

