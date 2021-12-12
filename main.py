import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

st.write("# Effect of train data size")

@st.cache
def get_data():
    df = pd.read_csv("fish.csv")
    return df
df = get_data()

show_data = st.checkbox("Show data")
if show_data:
    st.dataframe(df)

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

mape_df = pd.DataFrame(
    {
        "train_size": [i for i in range(10, 91, 10)],
        "mape": mape_lst
    }
)

avg = sum(mape_df["mape"]) / len(mape_df["mape"])
avg_df = pd.DataFrame(
    {
        'y': [avg]
    }
)

chart_style = st.selectbox("Chart type", ("Matplotlib", "Vega"))

if chart_style == "Matplotlib":
    plt.rcParams["figure.figsize"] = (10,5)
    bar = st.checkbox("Bar chart")
    if bar:
        plt.bar(mape_df["train_size"], mape_df["mape"], width=7)
        plt.plot([5, 95], [avg, avg], color="red")
        plt.ylim([0, 1])
        plt.legend(loc="upper right")
        plt.xlabel("Train size in %")
        plt.ylabel("MAPE")
        plt.grid(axis="y")
        plt.tight_layout()
    else:        
        plt.plot(mape_df["train_size"], mape_df["mape"], label="MAPE", marker="o")
        plt.plot([10, 90], [avg, avg], color="red")
        plt.grid()
        plt.ylim([0, 1])
        plt.legend(loc="upper right")
        plt.xlabel("Train size in %")
        plt.ylabel("MAPE")
        plt.tight_layout()
    st.pyplot(plt)
elif chart_style == "Vega":
    bar = st.checkbox("Bar chart")
    if bar:
        mape_chart = alt.Chart(mape_df).mark_bar(size=50).encode(
            x=alt.X('train_size',
                axis=alt.Axis(title='Train size')
                ),
            y=alt.Y('mape',
                axis=alt.Axis(title='MAPE'),
                scale=alt.Scale(domain=(0, 1))
                )
        )
        line = alt.Chart(avg_df).mark_rule(color="red").encode(y='y')
        st.altair_chart(mape_chart + line, use_container_width=True)
    else:
        mape_chart = alt.Chart(mape_df).mark_line(point=alt.OverlayMarkDef()).encode(
        x=alt.X('train_size',
                axis=alt.Axis(title='Train size')
                ),
        y=alt.Y('mape',
                axis=alt.Axis(title='MAPE'),
                scale=alt.Scale(domain=(0, 1))
                )
        )
        line = alt.Chart(avg_df).mark_rule(color="red").encode(y='y')
        st.altair_chart(mape_chart + line, use_container_width=True)