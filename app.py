import streamlit as st
import pickle
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# Teams and cities
teams = [
    'Chennai Super Kings',
    'Mumbai Indians',
    'Kolkata Knight Riders',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Punjab Kings',
    'Sunrisers Hyderabad',
    'Lucknow Super Giants',
    'Gujarat Titans',
    'Royal Challengers Bengaluru'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru', 'Navi Mumbai', 'Lucknow',
    'Guwahati', 'Dubai'
]

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the pickled model
        with open('ipl_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'ipl_model.pkl' not found. Please run 'train_model.py' first to generate the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Main app
def main():
    # Header
    st.title('🏏 IPL Win Predictor')
    st.markdown('---')
    st.markdown("""
    ### Predict match outcomes in real-time!
    This app predicts the probability of a team winning an IPL match based on current match conditions.
    """)
    st.markdown('---')

    # Load model
    model = load_model()

    # Team selection
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('🏏 Select the batting team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('⚾ Select the bowling team', sorted(teams))

    # Validation: Teams should be different
    if batting_team == bowling_team:
        st.error("⚠️ Batting and bowling teams must be different!")
        st.stop()

    # City selection
    selected_city = st.selectbox('🏟️ Select host city', sorted(cities))

    st.markdown('---')

    # Match details
    st.subheader('📊 Current Match Situation')

    target = st.number_input('🎯 Target Score', min_value=1, max_value=300, value=180, step=1)

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input('📈 Current Score', min_value=0, max_value=300, value=50, step=1)
    with col4:
        overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=19.5, value=10.0, step=0.1, format="%.1f")
    with col5:
        wickets = st.number_input('🚫 Wickets Lost', min_value=0, max_value=10, value=2, step=1)

    st.markdown('---')

    # Predict button
    if st.button('🔮 Predict Win Probability', use_container_width=True):
        # Validation
        if score >= target:
            st.success(f"🎉 {batting_team} has already won the match!")
            st.balloons()
            st.stop()

        if wickets >= 10:
            st.error(f"🏁 {batting_team} is all out! {bowling_team} wins!")
            st.stop()

        if overs >= 20:
            st.error(f"⏰ Innings completed! {bowling_team} wins!")
            st.stop()

        if overs == 0:
            st.warning("⚠️ Please enter a valid number of overs (greater than 0)")
            st.stop()

        # Calculate features
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Create input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Make prediction
        try:
            result = model.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            # Display results
            st.markdown('---')
            st.subheader('📊 Win Probability')

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    label=f"🏏 {batting_team}",
                    value=f"{round(win_prob * 100)}%",
                    delta="Batting"
                )
                st.progress(win_prob)

            with col_b:
                st.metric(
                    label=f"⚾ {bowling_team}",
                    value=f"{round(loss_prob * 100)}%",
                    delta="Bowling"
                )
                st.progress(loss_prob)

            st.markdown('---')

            # Additional match stats
            st.subheader('📈 Match Statistics')
            stat_col1, stat_col2, stat_col3 = st.columns(3)

            with stat_col1:
                st.metric("Runs Required", runs_left)
            with stat_col2:
                st.metric("Balls Remaining", balls_left)
            with stat_col3:
                st.metric("Wickets in Hand", wickets_left)

            stat_col4, stat_col5 = st.columns(2)
            with stat_col4:
                st.metric("Current Run Rate", f"{crr:.2f}")
            with stat_col5:
                st.metric("Required Run Rate", f"{rrr:.2f}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Footer
    st.markdown('---')
    st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ for Cricket Fans | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
