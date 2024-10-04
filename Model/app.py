import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model('model_tip_predictor.h5')
scaler = joblib.load('scaler.pkl')

# Title of the web app
st.title("Prediksi Tips Waiters dalam Rupiah")

# Input fields for the user
total_bill = st.number_input('Total Tagihan (Rupiah)', min_value=00.00, value=00.00)
gender = st.selectbox('Jenis Kelamin', ('Female', 'Male'))
smoker = st.selectbox('Perokok', ('No', 'Yes'))
day = st.selectbox('Hari', ('Thur', 'Fri', 'Sat', 'Sun'))
time = st.selectbox('Waktu', ('Lunch', 'Dinner'))
people = st.number_input('Jumlah Orang', min_value=1, value=1)

# Map categorical values to numerical (as done in training)
gender_map = {'Female': 0, 'Male': 1}
smoker_map = {'No': 0, 'Yes': 1}
day_map = {'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3}
time_map = {'Lunch': 0, 'Dinner': 1}

# Prepare the input for the model
input_data = np.array([[total_bill, gender_map[gender], smoker_map[smoker], 
                        day_map[day], time_map[time], people]])

# Scale the input data (using the saved scaler)
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button('Prediksi Tips'):
    prediction = model.predict(input_data_scaled)
    # Assuming the predicted tip is in Rupiah
    st.write(f"Prediksi Tips: Rp {prediction[0][0]:,.3f}")
