import pandas as pd
def load():
    data = pd.read_csv("data/dc_weather.csv")
    return data
