import streamlit as st
import pandas as pd

st.write("# Effect of train data size")

@st.cache
def get_data():
    df = pd.read_csv("fish.csv")
    return df
df = get_data()

@st.cache
def show_data_func():
    df_to_display = df.iloc[:5]
    return df_to_display

show_data = st.checkbox("Show data")
if show_data:
    st.table(show_data_func())

x = df[["Height", "Width", "Length"]]
y = df["Weight"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

@st.cache
def mape_list_func():
    mape_lst = []
    for train_size in range(1, 10):
        train_size = train_size / 10
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, train_size=train_size)
        m = LinearRegression()
        m.fit(train_x, train_y)
        p = m.predict(test_x)
        mape = mean_absolute_percentage_error(p, test_y)
        mape_lst.append(mape)
    return mape_lst

mape_lst = mape_list_func()

st.write(mape_lst)