import streamlit as st
import joblib

# Load the saved model
model = joblib.load('decision_tree_model.pkl')

# Define function to make predictions
def predict_booking(features):
    prediction = model.predict([features])
    return 'Canceled' if prediction == 1 else 'Not Canceled'

# Streamlit app interface
st.title("Hotel Booking Cancellation Prediction")

# Input fields for user to enter feature values
no_of_adults = st.number_input("Number of Adults", min_value=0, max_value=10, value=0)
no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=7, value=0)
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, max_value=7, value=0)
lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=0)
arrival_year = st.number_input("Arrival Year", min_value=2021, max_value=2025, value=2024)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=9)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=1)
no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, max_value=10, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, max_value=10, value=0)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, max_value=1000.0, value=0.0)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=0)
required_car_parking_space = st.selectbox("Required Car Parking Space", ["Yes", "No"])
required_car_parking_space_value = 1 if required_car_parking_space == "Yes" else 0
repeated_guest = st.selectbox("Repeated Guest", ["Yes", "No"])
repeated_guest_value = 1 if repeated_guest == "Yes" else 0

# Categorical features: encode based on how the model was trained
room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type_1', 'Room_Type_2', 'Room_Type_3', 'Room_Type_4', 'Room_Type_5', 'Room_Type_6', 'Room_Type_7'])
type_of_meal_plan = st.selectbox("Meal Plan", ['Not_Selected', 'Meal_Plan_1', 'Meal_Plan_2', 'Meal_Plan_3'])
market_segment_type = st.selectbox("Market Segment", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])

# Manually encode these features as per training
room_type_dict = {'Room_Type_1': 0, 'Room_Type_2': 1, 'Room_Type_3': 2,  'Room_Type_4':3, 'Room_Type_5':4, 'Room_Type_6':5, 'Room_Type_7':6}
meal_plan_dict = {'Not_Selected':0, 'Meal_Plan_1':1, 'Meal_Plan_2':2, 'Meal_Plan_3':3}
market_segment_dict = {'Offline': 0, 'Online': 1, 'Corporate': 2, 'Aviation': 3, 'Complementary': 4}

# Create a dictionary of inputs for prediction
inputs = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'lead_time': lead_time,
    'arrival_year': arrival_year,
    'arrival_month': arrival_month,
    'arrival_date': arrival_date,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests,
    'required_car_parking_space': required_car_parking_space_value,  # Use encoded value here
    'repeated_guest': repeated_guest_value,  # Use encoded value here
    'room_type_reserved': room_type_dict[room_type_reserved],
    'type_of_meal_plan': meal_plan_dict[type_of_meal_plan],
    'market_segment_type': market_segment_dict[market_segment_type]
}

# Combine all features for prediction
features = [
    no_of_adults,
    no_of_children,
    no_of_weekend_nights,
    no_of_week_nights,
    lead_time,
    arrival_year,
    arrival_month,
    arrival_date,
    no_of_previous_cancellations,
    no_of_previous_bookings_not_canceled,
    avg_price_per_room,
    no_of_special_requests,
    required_car_parking_space_value,  # Corrected to pass the encoded value
    repeated_guest_value,  # Corrected to pass the encoded value
    room_type_dict[room_type_reserved],  # Encoded value of Room Type
    meal_plan_dict[type_of_meal_plan],   # Encoded value of Meal Plan
    market_segment_dict[market_segment_type]  # Encoded value of Market Segment
]

# Display prediction button
if st.button('Predict Booking Cancellation'):
    result = predict_booking(features)
    st.success(f"The booking is predicted to be: {result}")
