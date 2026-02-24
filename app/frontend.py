import streamlit as st
import requests
from src.gemstones.exception import CustomException
from src.gemstones.logger import logging
import sys

st.title('Gemstones Price Predictor')

with st.form('Gem Features: '):
    carat = st.number_input('carat of gem', min_value = 0.0, max_value = 3.5, value = 1.0)
    cut = st.selectbox(label = 'cut quality of gem', options = ['Premium', 'Very Good', 'Ideal', 'Good', 'Fair'])
    color = st.selectbox(label = 'color of the gem', options = ['F', 'J', 'G', 'E', 'D', 'H', 'I'])
    clarity = st.selectbox(label = 'clarity of the gem', options = ['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1'])
    depth = st.number_input(label = 'depth of the gem', min_value = 0.0, max_value = 72.0, value = 35.0)
    table = st.number_input(label = 'tabel of the gem', min_value = 0.0, max_value = 80.0, value = 40.0)
    x = st.number_input(label = 'dimensions of gem (x)', min_value = 0.0, max_value = 10.0, value = 5.0)
    y = st.number_input(label = 'dimensions of gem (y)', min_value = 0.0, max_value = 10.0, value = 5.0)
    z = st.number_input(label = 'dimensions of gem (z)', min_value = 0.0, max_value = 35.0, value = 15.0)

    submit = st.form_submit_button(label = 'Submit')

if submit:
    paylod = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }

    try:
        with st.spinner('Finding result'):
            response = requests.post('http://127.0.0.1:8000/predict', json = paylod)

            if response.status_code == 200:
                prediction = response.json()
                value = response.json().get('predicted', [0])[0]
                st.success(f'Predicted rate is: ${round(value, 2)}')
                st.balloons()
    except Exception as e:
        logging.exception('Error occurred at frontend')
        raise CustomException(e, sys)