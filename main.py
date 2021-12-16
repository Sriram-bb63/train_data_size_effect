import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from streamlit.elements.map import _DEFAULT_MAP
plt.rcParams["figure.figsize"] = (10,5)

st.write("# Effect of train data size")

st.write("---")

@st.cache
def get_data():
    df = pd.read_csv("Fish.csv")
    return df
df = get_data()

show_data = st.checkbox("Show data")
if show_data:
    st.dataframe(df)

x = df[["Height", "Width", "Length"]]
y = df["Weight"]

st.write("---")

st.write("## Choose models")

decision_tree_checkbox = st.checkbox("Decision tree")
random_forest_checkbox = st.checkbox("Ridge")

from sklearn.linear_model import LinearRegression

models = {"Linear regression": LinearRegression()}

if decision_tree_checkbox:
    from sklearn.tree import DecisionTreeRegressor
    models["Decision tree"] = DecisionTreeRegressor()
if random_forest_checkbox:
    from sklearn.linear_model import Ridge
    models["Ridge"] = Ridge()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def trainer(models):
    for i in models.keys():
        model = models[i]
        model_mape_lst = []
        for train_size in range(10, 91, 10):
            train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, train_size=train_size)
            model.fit(train_x, train_y)
            p = model.predict(test_x)
            mape = mean_absolute_percentage_error(p, test_y)
            model_mape_lst.append(mape)
        models[i] = model_mape_lst
    return models

mape_dict = trainer(models)

mape_df = pd.DataFrame()

mape_df["Linear regression"] = mape_dict["Linear regression"]
if decision_tree_checkbox:
    mape_df["Decision tree"] = mape_dict["Decision tree"]
if random_forest_checkbox:
    mape_df["Ridge"] = mape_dict["Ridge"]

st.write("---")

show_mape_data = st.checkbox("MAPE df")

if show_mape_data:
    st.table(mape_df)

chart_style = st.selectbox("Chart style", ("Matplotlib", "Vega"))

average_checkbox = st.checkbox("Average MAPE")

if average_checkbox:
    avg_fig, ax = plt.subplots()
    avg_mape_dict = dict()
    for col in mape_df.columns:
        avg_mape = sum(mape_df[col]) / len(mape_df[col])
        avg_mape_dict[col] = avg_mape
    ax.bar(avg_mape_dict.keys(), avg_mape_dict.values())
    st.pyplot(avg_fig)


if chart_style == "Matplotlib":
    mape_fig, ax = plt.subplots()
    for col in mape_df.columns:
        ax.plot(mape_df.index*10, mape_df[col], marker=".", label=col)
    ax.legend(loc="upper right")
    ax.grid()
    plt.tight_layout()
    ax.set_xlabel("Train data size in %")
    ax.set_ylabel("MAPE")
    ax.set_ylim([0, 2])
    plt.style.use("seaborn")
    st.pyplot(mape_fig)


















# if chart_style == "Matplotlib":
#     plt.rcParams["figure.figsize"] = (10,5)
#     if bar:
#         plt.bar(mape_df["train_size"], mape_df["mape"], width=7)
#         plt.plot([5, 95], [avg, avg], color="red")
#         plt.ylim([0, 1])
#         plt.legend(loc="upper right")
#         plt.xlabel("Train size in %")
#         plt.ylabel("MAPE")
#         plt.grid(axis="y")
#         plt.tight_layout()
# elif chart_style == "Vega":
#     if bar:
#         mape_chart = alt.Chart(mape_df).mark_bar(size=50).encode(
#             x=alt.X('train_size',
#                 axis=alt.Axis(title='Train size in %')
#                 ),
#             y=alt.Y('mape',
#                 axis=alt.Axis(title='MAPE'),
#                 scale=alt.Scale(domain=(0, 1))
#                 )
#             )
#         line = alt.Chart(avg_df).mark_rule(color="red").encode(y='y')
#         st.altair_chart(mape_chart + line, use_container_width=True)
#     else:
#         mape_chart = alt.Chart(mape_df).mark_line(point=alt.OverlayMarkDef()).encode(
#         x=alt.X('train_size',
#                 axis=alt.Axis(title='Train size in %')
#                 ),
#         y=alt.Y('mape',
#                 axis=alt.Axis(title='MAPE'),
#                 scale=alt.Scale(domain=(0, 1))
#                 )
#             )
#         line = alt.Chart(avg_df).mark_rule(color="red").encode(y='y')
#         st.altair_chart(mape_chart + line, use_container_width=True)

smol_docs = """
## What is train size?

The amount of data which is used to train a model is called as train size. Very less training data or too much training data will affect the model's accuracy negatively.

## What is MAPE?

The average error of the predictions made by the model in percentage is called MAPE(Mean Absolute Percentage Error).
"""


st.sidebar.write(smol_docs)

st.sidebar.markdown("""---""")

links = """
Made by [Sriram Vasudevan](https://sriram-bb63.github.io/)

Source code [Github](https://github.com/Sriram-bb63/train_data_size_effect)
"""

st.sidebar.write(links)











# def trainer(models, mape_lst):
#     for model in models:
#         model_mape_lst = []
#         for train_size in range(10, 91, 10):
#             train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, train_size=train_size)
#             model.fit(train_x, train_y)
#             p = model.predict(test_x)
#             mape = mean_absolute_percentage_error(p, test_y)
#             model_mape_lst.append(mape)
#         mape_lst.append(model_mape_lst)
#     return mape_lst