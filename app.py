import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

with open('logistic_regression_model.pkl', 'rb') as file:
    log_model = pickle.load(file)


DATASET_PATH = "data/data.zip/data_cleaned_final.csv"
LOG_MODEL_PATH = "logistic_regression_model.pkl"

def main():
    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pd.read_csv(DATASET_PATH)
        return heart_df

    def user_input_features() -> pd.DataFrame:
        age_cat_options = ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                   "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                   "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                   "Age 80 or older"]
        age_cat = st.sidebar.selectbox("Age category", options=age_cat_options)      
        smoking_cat_options = ["Never smoked", "Former smoker", "Current smoker (Smokes everyday)", "Current smoker (Smokes somedays)"]
        smoking = st.sidebar.selectbox("What is your current smoking status", options=smoking_cat_options)
        stroke = st.sidebar.selectbox("Have you ever experienced a stroke?", options=("No", "Yes"))
        lung = st.sidebar.selectbox("Do you have a history of lung disease", options=("No", "Yes"))
        jointpain = st.sidebar.selectbox("Do you have a history of joint pain?", options=("No", "Yes"))
        kidney = st.sidebar.selectbox("Do you have a history of kidney problems?", options=("No", "Yes"))
        hearing = st.sidebar.selectbox("Do you have any hearing impairment or deafness?", options=("No", "Yes"))
        chestpain = st.sidebar.selectbox("Do you have a history of chest pain?", options=("No", "Yes"))
        

        features = pd.DataFrame({
            "AgeCategory": [age_cat],
            "SmokerStatus": [smoking],
            "HadStroke": [stroke],
            "HadCOPD": [lung],
            "HadArthritis": [jointpain],
            "HadKidneyDisease": [kidney],
            "HadKidneyDisease": [hearing],
            "HadAngina": [chestpain],
        })

        return features

    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart-fav.png"
    )

    st.title("Web Application in Predicting Heart Disease")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor_new.png",
                 caption="Let's diagnose your heart health!",
                 width=150)
        st.markdown('##')
        m = st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #00cc00;color:white;
            font-size:15px;
            height:4em;
            width:20em;
            margin-top: 8cm; 
            }
            </style>""", unsafe_allow_html=True)
        submit = st.button("Predict")

    with col2:
        st.markdown("""
        **Did you know that heart attacks are the leading cause of death globally?**
        
        Heart disease is a leading cause of mortality worldwide, imposing a significant burden on healthcare systems and individuals.   Early detection and prevention are essential to mitigate its impact. As of the 2022 update, the demand for accurate predictive tools to assess heart disease risk is growing.
        
        In response, we propose the "Heart Disease Prediction" project, which utilizes data analytics and machine learning techniques to develop a predictive system. This system aims to offer timely risk assessments, personalized interventions, and address the pressing need for proactive measures in preventing heart disease.
        
        Here are a few steps away to predicting your heart disease status:
            
        1. Enter the parameters based on your health status.
        
        2. Press the "Predict" button and wait seconds to get the result. 

        Please remember that this result does not constitute a diagnosis from a doctor! Healthcare facilities would never employ it. 
        Therefore, if you experience any concerns or problems, see a doctor.
        
        
            """)

    heart = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=200)

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=1)

    df.fillna(0, inplace=True)  # Handle missing values more thoughtfully based on your model

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
                                   
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
           
                        

if __name__ == "__main__":
    main()
