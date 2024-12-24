
import streamlit as st
import numpy as np
import joblib

# Load the trained KMeans model and scaler
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define input fields for customer details
st.title("Customer Cluster Prediction")

# Input fields
avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, value=100.0, step=0.1)
lead_time = st.number_input('Lead Time (days)', min_value=0, value=50)
no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=1)
no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, value=0)
repeated_guest = st.selectbox('Is Repeated Guest?', [0, 1])
market_segment_type = st.selectbox('Market Segment Type', [0, 1, 2, 3])  # Assuming encoded market segments

# Collect the input data into a numpy array
customer_data = np.array([[avg_price_per_room, lead_time, no_of_special_requests,
                           no_of_previous_bookings_not_canceled, repeated_guest, market_segment_type]])

# Button to predict the cluster
if st.button('Predict Cluster'):
    # Scale the input data
    customer_data_scaled = scaler.transform(customer_data)

    # Predict the cluster
    cluster = kmeans_model.predict(customer_data_scaled)[0]

    # Assign cluster labels
    if cluster == 0:
        cluster_label = 'Budget'
    elif cluster == 1:
        cluster_label = 'Luxury'
    else:
        cluster_label = 'Frequent'

    # Display the predicted cluster
    st.write(f'The predicted customer cluster is: **{cluster_label}**')
