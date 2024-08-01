import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Non Diabetic'
    else:
        return 'Diabetic'

def main():
    Pregnancies = st.text_input('Pregnancies:')
    Glucose = st.text_input('Glucose:')
    BloodPressure = st.text_input('Blood Pressure:')
    SkinThickness = st.text_input('Skin Thickness:')
    Insulin = st.text_input('Insulin:')
    BMI = st.text_input('BMI:')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function:')
    Age = st.text_input('Age:')

    diagnosis = ''

    if st.button('Predict'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()