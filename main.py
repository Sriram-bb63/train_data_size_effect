import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

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

st.write("## Choose models")

# linear_regression_checkbox = st.checkbox("Linear regression")
decision_tree_checkbox = st.checkbox("Decision tree")
random_forest_checkbox = st.checkbox("Random forest")

from sklearn.linear_model import LinearRegression

models = [LinearRegression()]

# if linear_regression_checkbox:
#     from sklearn.linear_model import LinearRegression
#     models.append("LinearRegression")
if decision_tree_checkbox:
    from sklearn.tree import DecisionTreeRegressor
    models.append(DecisionTreeRegressor())
if random_forest_checkbox:
    from sklearn.linear_model import Ridge
    models.append(Ridge())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def trainer(models, mape_lst):
    for model in models:
        model_mape_lst = []
        for train_size in range(10, 91, 10):
            train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, train_size=train_size)
            model.fit(train_x, train_y)
            p = model.predict(test_x)
            mape = mean_absolute_percentage_error(p, test_y)
            model_mape_lst.append(mape)
        mape_lst.append(model_mape_lst)
    return mape_lst

mape_lst = trainer(models, [])

mape_df = pd.DataFrame()

for i in range(len(mape_lst)):
    mape_df[f"model_{i+1}"] = mape_lst[i]

st.table(mape_df)

# mape_df = pd.DataFrame(
#     {
#         "train_size": [i for i in range(10, 91, 10)],
#         "mape": mape_lst
#     }
# )

# avg = sum(mape_df["mape"]) / len(mape_df["mape"])
# avg_df = pd.DataFrame(
#     {
#         'y': [avg]
#     }
# )

# chart_style = st.selectbox("Chart type", ("Matplotlib", "Vega"))

# bar = st.checkbox("Bar chart")

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
#     else:        
#         plt.plot(mape_df["train_size"], mape_df["mape"], label="MAPE", marker="o")
#         plt.plot([10, 90], [avg, avg], color="red")
#         plt.grid()
#         plt.ylim([0, 1])
#         plt.legend(loc="upper right")
#         plt.xlabel("Train size in %")
#         plt.ylabel("MAPE")
#         plt.tight_layout()
#     st.pyplot(plt)
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