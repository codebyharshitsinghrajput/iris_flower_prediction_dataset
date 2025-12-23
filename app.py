import streamlit as st
import numpy as np
import pickle

with open("iris_dataset.pkl",'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")

sepal_length=st.slider("sepal lenght(cm)",4.3,7.9)
sepal_width=st.slider("sepal width(cm)",2.0,4.4)
petal_length=st.slider("petal lenght(cm)",1.0,6.9)
petal_width=st.slider("petal width(cm)",0.1,2.5)

if st.button("Predict"):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction =model.predict(input_data)
    species=['Setosa','Versicolor','Virginica']
    st.success(f"Predicted Iris Species:{species[prediction[0]]}")
    
