import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("ipl_model.pkl")

# Team list
teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

st.title("🏏 IPL Score Predictor")
st.write("Predict the final score based on match situation!")

# User inputs
batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)

if batting_team == bowling_team:
    st.warning("Batting and bowling team cannot be the same!")

overs = st.number_input("Overs completed", min_value=5.0, max_value=20.0, step=0.1)
runs = st.number_input("Current runs", min_value=0)
wickets = st.number_input("Current wickets", min_value=0, max_value=10)
runs_in_prev_5 = st.number_input("Runs in last 5 overs", min_value=0)
wickets_in_prev_5 = st.number_input("Wickets in last 5 overs", min_value=0)

# Prediction function
def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    temp_array = []

    # One-hot encoding for batting team
    for team in teams:
        temp_array.append(1 if batting_team == team else 0)

    # One-hot encoding for bowling team
    for team in teams:
        temp_array.append(1 if bowling_team == team else 0)

    # Add match stats
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])

    # Convert to numpy array
    temp_array = np.array([temp_array])

    # Predict score
    predicted_score = int(model.predict(temp_array)[0])
    return predicted_score

# Button to trigger prediction
if st.button("Predict Score"):
    if batting_team != bowling_team:
        score = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5)
        st.success(f"Predicted Final Score: {score-10} to {score+5}")
