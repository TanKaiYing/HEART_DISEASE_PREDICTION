import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    log_model = pickle.load(model_file)

# Function to encode user inputs
def encode_input(input_data):
    # Encoding mappings
    age_category_map = {"Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3,
                        "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7,
                        "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10, "Age 75 to 79": 11,
                        "Age 80 or older": 12}
    smoking_status_map = {"Never smoked": 0, "Former smoker": 1, "Current smoker (Smokes everyday)": 2, "Current smoker (Smokes somedays)": 3}
    yes_no_map = {"No": 0, "Yes": 1}

    # Apply mappings
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_map)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoking_status_map)
    input_data['HadStroke'] = input_data['HadStroke'].map(yes_no_map)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(yes_no_map)
    input_data['HadArthritis'] = input_data['HadArthritis'].map(yes_no_map)
    input_data['HadKidneyDisease'] = input_data['HadKidneyDisease'].map(yes_no_map)
    input_data['HadAngina'] = input_data['HadAngina'].map(yes_no_map)

    # need to change
    # input_data['PhysicalHealthDays'] = input_data['PhysicalHealthDays'].map(yes_no_map)
    # input_data['MentalHealthDays'] = input_data['MentalHealthDays'].map(yes_no_map)
    # input_data['HeightInMeters'] = input_data['HeightInMeters'].map(yes_no_map)
    # input_data['WeightInKilograms'] = input_data['WeightInKilograms'].map(yes_no_map)
    # input_data['BMI'] = input_data['BMI'].map(yes_no_map)
    # input_data['GeneralHealth'] = input_data['GeneralHealth'].map(yes_no_map)
    # input_data['RemovedTeeth'] = input_data['RemovedTeeth'].map(yes_no_map)
    # input_data['ChestScan'] = input_data['ChestScan'].map(yes_no_map)
    # input_data['DifficultyWalking'] = input_data['DifficultyWalking'].map(yes_no_map)
    # input_data['HadDiabetes'] = input_data['HadDiabetes'].map(yes_no_map)
    # input_data['PneumoVaxEver'] = input_data['PneumoVaxEver'].map(yes_no_map)
    # input_data['DeafOrHardOfHearing'] = input_data['DeafOrHardOfHearing'].map(yes_no_map)



    return input_data

def main():
    st.set_page_config(page_title="Heart Disease Prediction App", page_icon=":heart:")

    st.title("Heart Disease Prediction App")
    st.subheader("Evaluate your risk of heart disease.")

    # User input
    age_cat_options = ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                       "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                       "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                       "Age 80 or older"]
    smoking_cat_options = ["Never smoked", "Former smoker", "Current smoker (Smokes everyday)", "Current smoker (Smokes somedays)"]
    yes_no_options = ["No", "Yes"]

    # Create a sidebar for input
    age_cat = st.sidebar.selectbox("Age category", options=age_cat_options)      
    smoking = st.sidebar.selectbox("What is your current smoking status", options=smoking_cat_options)
    stroke = st.sidebar.selectbox("Have you ever experienced a stroke?", options=yes_no_options)
    lung = st.sidebar.selectbox("Do you have a history of lung disease?", options=yes_no_options)
    jointpain = st.sidebar.selectbox("Do you have a history of joint pain?", options=yes_no_options)
    kidney = st.sidebar.selectbox("Do you have a history of kidney problems?", options=yes_no_options)
    chestpain = st.sidebar.selectbox("Do you have a history of chest pain?", options=yes_no_options)

    # Compile user inputs into a DataFrame
    input_data = pd.DataFrame({
        "AgeCategory": [age_cat],
        "SmokerStatus": [smoking],
        "HadStroke": [stroke],
        "HadCOPD": [lung],
        "HadArthritis": [jointpain],
        "HadKidneyDisease": [kidney],
        "HadAngina": [chestpain],
    })

    # Encode user inputs
    input_encoded = encode_input(input_data)

    # Scale user inputs
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_encoded)

    # Prediction button
    submit = st.sidebar.button("Predict")

    if submit:
        # Make a prediction
        prediction = log_model.predict(input_scaled)
        prediction_prob = log_model.predict_proba(input_scaled)

        # Display results
        st.subheader('Prediction Results')
        if prediction[0] == 0:
            st.success(f"Low risk of heart disease. Probability: {prediction_prob[0][0] * 100:.2f}%")
        else:
            st.error(f"High risk of heart disease. Probability: {prediction_prob[0][1] * 100:.2f}%")

if __name__ == "__main__":
    main()



