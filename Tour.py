# import the libraries
import streamlit as st
import keras
from PIL import Image
import numpy as np
from utilities import get_scaler

## loading the ann model
model = keras.models.load_model("complete_model.h5")
scaler = get_scaler()


## create a function for prediction
def Tourist_prediction(month, year):
    input_array = np.array([month, year])  # Create a NumPy array from the input
    input_reshape = input_array.reshape(1, -1)
    prediction = model.predict(input_reshape)
    return prediction


def main():
    ## set page configuration
    st.set_page_config(page_title="Tourist Predictor", layout="wide")

    ## add image
    image = Image.open("tourist.jpeg")
    st.image(image, use_column_width=True)

    ## add page title and content
    st.title("Tourism Tracker Using Artificial Neural Networks")
    st.write("Enter your personalised data to predict tourist numbers in a location")

    ## variable inputs
    month = st.number_input(
        "Month of travel(1 for January, 2 for February etc.):",
        min_value=1,
        max_value=12,
        step=1,
    )
    year = st.number_input("Year of travel:", min_value=1, step=1, value=2023)

    ## code for prediction
    if st.button("Predict"):
        prediction = Tourist_prediction(month, year)
        st.success(
            f"The likely number of tourists is expected to be: {prediction[0][0]*1000:.2f}"
        )  # Display the prediction


if __name__ == "__main__":
    main()
