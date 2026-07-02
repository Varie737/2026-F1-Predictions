
# imports
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut

# Austria Quali
austria_quali= {
    "RUS":(16.424000, 29.620000, 20.069000, 66.113, 1),
    "LEC":(16.485000, 29.632000, 20.232000, 66.349, 2),
    "HAM":(16.521000, 29.690000, 20.197000, 66.408, 3),
    "ANT":(16.564000, 29.852000, 19.998000, 66.414, 4),
    "VER":(16.501000, 29.719000, 20.255000, 66.475, 5),
    "NOR":(16.576000, 29.642000, 20.284000, 66.502, 6),
    "PIA":(16.636000, 29.742000, 20.133000, 66.511, 7),
    "HAD":(16.717000, 29.787000, 20.128000, 66.632, 8),
    "LAW":(16.693000, 29.937000, 20.325000, 67.955, 9),
    "LIN":(16.592000, 29.992000, 20.423000, 66.955, 10),
    "GAS":(16.716000, 30.001000, 20.506000, 67.223, 11),
    "BOR":(16.749000, 30.036000, 20.508000, 67.293, 12),
    "BEA":(16.759000, 30.118000, 20.646000, 67.523, 13),
    "HUL":(16.866000, 30.170000, 20.575000, 67.611, 14),
    "OCO":(16.804000, 30.341000, 20.672000, 67.817, 15),
    "COL":(16.692000, 30.421000, 20.781000, 68.171, 16), 
    "SAI":(16.845000, 30.504000, 20.903000, 68.252, 17),
    "ALB":(16.854000, 30.625000, 21.030000, 68.509, 18),
    "PER":(16.941000, 30.827000, 21.177000, 68.945, 19),
    "BOT":(16.928000, 30.725000, 21.377000, 69.030, 20),
    "ALO":(17.188000, 31.165000, 21.589000, 69.942, 21),
    "STR":(17.362000, 31.311000, 21.690000, 70.363, 22)
}

df = pd.DataFrame.from_dict(
    austria_quali, orient= "index" ,
    columns=["S1", "S2", "S3", "AustriaQuali_s", "AustriaGrid"]
)

df.index.name= "Driver"
df = df.reset_index()

df["UltimateLap_s"]= df["S1"] + df["S2"] + df["S3"]
pole = df["AustriaQuali_s"].min()
df["AustriaGapFromPole_s"] = (df["AustriaQuali_s"] - pole).round(3)


# Past GPs this season for learning since new regulations shift past years gp too much
# Australia
aus_grid= {
    "RUS": 1, "HAM": 7, "ANT": 2, "NOR": 6, "VER": 5, "HAD":3, "PIA": 5, "LAW": 8, "HUL":11, "LEC": 4,
    "LIN": 9, "BOR": 10, "COL": 16, "GAS": 14, "BEA": 12, 
    "SAI": 20, "OCO": 13, "ALB": 15, "PER": 18, "BOT": 19, "STR": 20, "ALO": 17
}

df["AustraliaGrid"]= df["Driver"].map(aus_grid)

# I also made all the dnfs the same position since it wasn't really determined on race pace
aus_finish_pos = {
    "RUS": 1, "HAM": 4, "ANT": 2, "NOR": 5, "VER": 6, "HAD":17, "PIA": 17, "LAW": 13, "HUL":17, "LEC": 3,
    "LIN": 8, "BOR": 9, "COL": 14, "GAS": 10, "BEA": 7, 
    "SAI": 15, "OCO": 11, "ALB": 13, "PER": 16, "BOT": 17, "STR": 17, "ALO": 17
}

df["AustraliaFinishPos"]= df["Driver"].map(aus_finish_pos)

# China
chi_grid= {
    "RUS": 2, "HAM": 3, "ANT": 1, "NOR": 6, "VER": 8, "HAD":9, "PIA": 5, "LAW": 14, "HUL":11, "LEC": 4,
    "LIN": 15, "BOR": 16, "COL": 12, "GAS": 7, "BEA": 10, 
    "SAI": 17, "OCO": 13, "ALB": 18, "PER": 22, "BOT": 20, "STR": 21, "ALO": 19
}

df["ChinaGrid"]= df["Driver"].map(chi_grid)

# I also made all the dnfs the same position since it wasn't really determined on race pace
chi_finish_pos = {
    "RUS": 2, "HAM": 3, "ANT": 1, "NOR": 16, "VER": 16, "HAD":8, "PIA": 16, "LAW": 7, "HUL":11, "LEC": 4,
    "LIN": 12, "BOR": 16, "COL": 10, "GAS": 6, "BEA": 5, 
    "SAI": 9, "OCO": 14, "ALB": 16, "PER": 15, "BOT": 13, "STR": 16, "ALO": 16
}

df["ChinaFinishPos"]= df["Driver"].map(chi_finish_pos)

# Japan
jap_grid= {
    "RUS": 2, "HAM": 6, "ANT": 1, "NOR": 5, "VER": 11, "HAD":8, "PIA": 3, "LAW": 14, "HUL":13, "LEC": 4,
    "LIN": 10, "BOR": 9, "COL": 15, "GAS": 7, "BEA": 18, 
    "SAI": 16, "OCO": 12, "ALB": 17, "PER": 19, "BOT": 20, "STR": 22, "ALO": 21
}

df["JapanGrid"]= df["Driver"].map(jap_grid)

# I also made all the dnfs the same position since it wasn't really determined on race pace
jap_finish_pos = {
    "RUS": 4, "HAM": 6, "ANT": 1, "NOR": 5, "VER": 8, "HAD":12, "PIA": 2, "LAW": 9, "HUL":11, "LEC": 3,
    "LIN": 14, "BOR": 13, "COL": 16, "GAS": 7, "BEA": 21, 
    "SAI": 15, "OCO": 10, "ALB": 20, "PER": 17, "BOT": 19, "STR": 21, "ALO": 18
}

df["JapanFinishPos"]= df["Driver"].map(jap_finish_pos)

# Miami
mia_grid= {
    "RUS": 5, "HAM": 6, "ANT": 1, "NOR": 4, "VER": 2, "HAD":22, "PIA": 7, "LAW": 11, "HUL":10, "LEC": 3,
    "LIN": 16, "BOR": 21, "COL": 8, "GAS": 9, "BEA": 12, 
    "SAI": 13, "OCO": 14, "ALB": 15, "PER": 20, "BOT": 19, "STR": 18, "ALO": 17
}

df["MiamiGrid"]= df["Driver"].map(mia_grid)

# PLACED KIMI AT 2ND B4 PENALTY WAS APPLIED TO FOCUS ON RACE PACE
# I also made all the dnfs the same position since it wasn't really determined on race pace
mia_finish_pos = {
    "RUS": 4, "HAM": 6, "ANT": 1, "NOR": 2, "VER": 5, "HAD":19, "PIA": 3, "LAW": 19, "HUL":19, "LEC": 8,
    "LIN": 14, "BOR": 12, "COL": 7, "GAS": 19, "BEA": 11, 
    "SAI": 9, "OCO": 13, "ALB": 10, "PER": 16, "BOT": 18, "STR": 17, "ALO": 15
}

df["MiamiFinishPos"]= df["Driver"].map(mia_finish_pos)

# Canada
can_grid= {
    "RUS": 1, "HAM": 5, "ANT": 3, "NOR": 3, "VER": 6, "HAD":7, "PIA": 4, "LAW": 12, "HUL":11, "LEC": 8,
    "LIN": 9, "BOR": 13, "COL": 10, "GAS": 14, "BEA": 16, 
    "SAI": 15, "OCO": 17, "ALB": 18, "PER": 20, "BOT": 22, "STR": 21, "ALO": 19
}

df["CanadaGrid"]= df["Driver"].map(can_grid)

# I also made all the dnfs the same position since it wasn't really determined on race pace
can_finish_pos = {
    "RUS": 17, "HAM": 2, "ANT": 1, "NOR": 17, "VER": 3, "HAD":5, "PIA": 11, "LAW": 7, "HUL":12, "LEC": 4,
    "LIN": 17, "BOR": 13, "COL": 6, "GAS": 8, "BEA": 10, 
    "SAI": 9, "OCO": 14, "ALB": 17, "PER": 17, "BOT": 16, "STR": 15, "ALO": 17
}

df["CanadaFinishPos"]= df["Driver"].map(can_finish_pos)

# Monaco
mon_grid= {
    "RUS": 6, "HAM": 3, "ANT": 1, "NOR": 8, "VER": 2, "HAD":5, "PIA": 7, "LAW": 10, "HUL":13, "LEC": 4,
    "LIN": 15, "BOR": 16, "COL": 14, "GAS": 9, "BEA": 19, 
    "SAI": 12, "OCO": 17, "ALB": 11, "PER": 18, "BOT": 20, "STR": 22, "ALO": 21
}

df["MonacoGrid"]= df["Driver"].map(mon_grid)

# PLACED KIMI AT 2ND B4 PENALTY WAS APPLIED TO FOCUS ON RACE PACE
# I also made all the dnfs the same position since it wasn't really determined on race pace
mon_finish_pos= {
    "RUS": 12, "HAM": 2, "ANT": 1, "NOR": 17, "VER": 17, "HAD":4, "PIA": 5, "LAW": 6, "HUL":13, "LEC": 17,
    "LIN": 7, "BOR": 11, "COL": 14, "GAS": 3, "BEA": 17, 
    "SAI": 16, "OCO": 9, "ALB": 8, "PER": 15, "BOT": 17, "STR": 17, "ALO": 10
}

df["MonacoFinishPos"]= df["Driver"].map(mon_finish_pos)

# Barcelona
barc_grid= {
    "RUS": 1, "HAM": 2, "ANT": 3, "NOR": 4, "VER": 5, "HAD":6, "PIA": 7, "LAW": 8, "HUL":9, "LEC": 10,
    "LIN": 11, "BOR": 12, "COL": 13, "GAS": 14, "BEA": 15, 
    "SAI": 16, "OCO": 17, "ALB": 18, "PER": 19, "BOT": 20, "STR": 21, "ALO": 22
}

df["BarcelonaGrid"]= df["Driver"].map(barc_grid)

# PLACED KIMI AT 2ND B4 PENALTY WAS APPLIED TO FOCUS ON RACE PACE
# I also made all the dnfs the same position since it wasn't really determined on race pace
barc_finish_pos = {
    "RUS": 3, "HAM": 1, "ANT": 2, "NOR": 4, "VER": 5, "HAD":7, "PIA": 6, "LAW": 9, "HUL":16, "LEC": 16,
    "LIN": 9, "BOR": 12, "COL": 11, "GAS": 8, "BEA": 16, 
    "SAI": 13, "OCO": 14, "ALB": 16, "PER": 15, "BOT": 16, "STR": 16, "ALO": 16
}

df["BarcelonaFinishPos"]= df["Driver"].map(barc_finish_pos)

# add constructor's data
team_points = {
    "McLaren": 141, "Mercedes": 262, "Red Bull": 89, "Williams": 11, "Ferrari": 190,
    "Haas": 21, "Aston Martin": 1, "Audi": 2, "Racing Bulls": 41, "Alpine": 57, "Cadillac": 0
}
max_pts = max(team_points.values())
team_score = {t: max(p, 0.5)/ max_pts for t, p in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "HAD": "Red Bull",
    "SAI": "Williams", "HUL": "Audi", "OCO": "Haas", "STR": "Aston Martin", 
    "ANT": "Mercedes", "LAW": "Racing Bulls", "BEA": "Haas", "COL": "Alpine", "LIN": "Racing Bulls",
    "ALB": "Williams", "BOR": "Audi", "BOT":"Cadillac", "PER": "Cadillac"
}

df["Team"]= df["Driver"].map(driver_to_team)
df["TeamScore"]= df["Team"].map(team_score)

# get weather data for austria
API_KEY = "d1f21ead9c466aebbc8e59d31d23f7bd"
weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat=47.22&lon=14.79&units=metric&appid={API_KEY}"

response = requests.get(weather_url)
weather_data = response.json()

# Check for successful API response before processing
if response.status_code == 200:
    forecast_time = "2026-06-28 12:00:00"
    forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

    # OpenWeather provides "pop" (Probability of Precipitation) between 0 and 1. 
    # Multiplying by 100 converts it to a standard 0-100% percentage scale.
    rain_probability = (forecast_data["pop"] * 100) if forecast_data else 0
    temperature = forecast_data["main"]["temp"] if forecast_data else 20

# adjust qualifying time based on weather conditions
if rain_probability >= 75:
    df["AustriaQuali_s"] = df["AustriaQuali_s"] * df["WetPerformanceFactor"]
else:
    df["AustriaQuali_s"] = df["AustriaQuali_s"]

df["Temperature"] = temperature
df["RainProbability"] = rain_probability

aus_pos_norm = (df["AustraliaFinishPos"]-1)/21
chi_pos_norm = (df["ChinaFinishPos"]-1)/21
jap_pos_norm = (df["JapanFinishPos"]-1)/21
mia_pos_norm = (df["MiamiFinishPos"]-1)/21
can_pos_norm = (df["CanadaFinishPos"]-1)/21
mon_pos_norm = (df["MonacoFinishPos"]-1)/21
barc_pos_norm = (df["BarcelonaFinishPos"]-1)/21

n= 7 # Number of races
df["RaceScore"] = ((1/n)*barc_pos_norm) + ((1/n)*mon_pos_norm) + ((1/n)*can_pos_norm) + ((1/n)*mia_pos_norm)  + ((1/n)*jap_pos_norm) 
+ ((1/n)*chi_pos_norm) + ((1/n)*aus_pos_norm)  

feature_cols = [
    "UltimateLap_s",
    "AustriaGapFromPole_s",
    "AustriaGrid",
    "TeamScore",
    "RainProbability",
    "Temperature"
]

X = df[feature_cols].copy()
y= df["RaceScore"]

imputer= SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# gradient boosting model
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree= 0.8, reg_lambda = 1.5, random_state=38)

loo = LeaveOneOut()
loo_errors = []
for train_idx, test_idx in loo.split(X_imputed):
    X_tr, X_te = X_imputed[train_idx], X_imputed[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_tr, y_tr)
    loo_errors.append(abs(model.predict(X_te)[0]-y_te.iloc[0]))

loo_mae= np.mean(loo_errors)

model.fit(X_imputed, y)
df["PredictedScore"]= model.predict(X_imputed)

final=df.sort_values("PredictedScore").reset_index(drop=True)

print("\n" + "="*68)
print("PREDICTED FINISHNG ORDER - 2026 AUSTRIA GP")
print("="*68)
print(final)
print(f"\nLeave One Out MAE: {loo_mae:.4f}")
