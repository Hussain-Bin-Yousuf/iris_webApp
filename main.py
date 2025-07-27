import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# 🌸 Simple Iris Flower Prediction App

This app predicts the **type of Iris flower** based on user input features.
""")

# Sidebar input section
st.sidebar.header('User Input Parameters')

def user_input_parameters():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 8.0, 5.4)
    sepal_width  = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width  = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width':  sepal_width,
        'petal_length': petal_length,
        'petal_width':  petal_width
    }
    return pd.DataFrame(data, index=[0])

df = user_input_parameters()

# Show user input
st.subheader("User Input Parameters")
st.dataframe(df)

# Load and train model
iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

# Make prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Output results
st.subheader("Class labels and their indices")
st.write(dict(enumerate(iris.target_names)))

st.subheader("Prediction")
st.success(f"Predicted Flower: {iris.target_names[prediction[0]]}")

st.subheader("Prediction Probability")
st.write(prediction_proba)
