import pandas as pd
import pickle


with open("td_short.pkl", "rb") as fp:
    df = pickle.load(fp)

print(df)
