import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load and cache the dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load data
df, target_names = load_data()

# Train the model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Streamlit UI
st.title("Iris Flower Species Prediction")
st.sidebar.title("Input Features")

# Sliders for user input
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()), float(df.iloc[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()), float(df.iloc[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()), float(df.iloc[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()), float(df.iloc[:, 3].mean()))

# Prediction
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Output
st.subheader("Prediction")
st.write(f"Predicted Species: {target_names[prediction[0]]}")

st.subheader("Prediction Probability")
st.bar_chart(prediction_proba[0])
