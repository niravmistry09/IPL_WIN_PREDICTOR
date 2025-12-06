import streamlit as st
import pickle
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI - Modern Colors
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: white !important;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .info-box h3 {
        color: #2d3748 !important;
        margin-top: 0;
    }
    .info-box p {
        color: #4a5568 !important;
    }
    .scenario-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
    }
    .scenario-box h3, .scenario-box h4 {
        color: #744210 !important;
        margin-top: 0;
    }
    .scenario-box p {
        color: #744210 !important;
    }
    .team-batting {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
    }
    .team-batting h3 {
        color: #1a365d !important;
    }
    .team-bowling {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
    }
    .team-bowling h3 {
        color: #742a2a !important;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    .help-text {
        font-size: 0.9rem;
        color: #4b5563 !important;
        font-style: italic;
    }
    /* Force dark text in all boxes */
    div[class*="box"] {
        color: #1f2937 !important;
    }
    div[class*="box"] strong {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

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
        with open('ipl_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'ipl_model.pkl' not found. Please run 'train_model.py' first to generate the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Beautiful Help Dialog - Using Native Streamlit Components
@st.dialog("📖 How to Use", width="large")
def show_help_dialog():
    # Welcome Header
    st.success("🏏 **Welcome to IPL Win Predictor!**")
    st.caption("Find out which team will win during the match!")

    st.markdown("---")

    # Understanding Section
    st.subheader("🎯 How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**1️⃣ First Innings Complete**  \nOne team batted and set a target")

    with col2:
        st.info("**2️⃣ Second Innings In Progress**  \nOther team is chasing now")

    st.markdown("---")

    # Step by Step Guide
    st.subheader("📝 Simple Steps")

    steps = [
        ("1", "Select the team that is **batting now**"),
        ("2", "Select the team that **batted first**"),
        ("3", "Choose the **stadium location**"),
        ("4", "Enter the **target runs** to win"),
        ("5", "Enter **current score, overs, and wickets**"),
        ("6", "Click **WHO WILL WIN?** button")
    ]

    for num, desc in steps:
        st.markdown(f"**Step {num}:** {desc}")

    st.markdown("---")

    # Example
    st.subheader("💡 Example")

    st.warning("**Match Scenario:**")
    st.markdown("""
    - 🔴 **Mumbai Indians** batted first → scored **180 runs**
    - 🔵 **Chennai Super Kings** is chasing → needs **181 runs**
    - 📊 CSK's current score: **95/2** in **12 overs**
    """)

    st.success("**What to enter:**")

    # Create a simple data display
    example_data = {
        "Field": ["Team Chasing", "Team Defending", "Venue", "Target", "Current Score", "Overs", "Wickets"],
        "Enter": ["Chennai Super Kings", "Mumbai Indians", "Chennai", "181", "95", "12.0", "2"]
    }

    df = pd.DataFrame(example_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.info("**Result:** App shows win chances for both teams!")

    st.markdown("---")

    # Quick Tips
    st.subheader("⚡ Quick Tips")

    tip_col1, tip_col2 = st.columns(2)

    with tip_col1:
        st.success("✅ **Chasing Team**  \n= Batting now")
        st.info("✅ **Target**  \n= Runs needed to win")

    with tip_col2:
        st.warning("✅ **Defending Team**  \n= Batted first")
        st.error("✅ **Wickets**  \n= Players out")

    # Action Buttons
    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🚀 Got It!", use_container_width=True, type="primary"):
            st.session_state.show_help = False
            st.rerun()

    with col_btn2:
        if st.button("❌ Close", use_container_width=True):
            st.session_state.show_help = False
            st.rerun()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 IPL Win Probability Predictor</h1>
        <p>Get real-time win predictions during the 2nd innings chase!</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()

    # Help button in header
    col_header1, col_header2 = st.columns([5, 1])
    with col_header2:
        if st.button("❓ How to Use", use_container_width=True, type="secondary"):
            st.session_state.show_help = True

    # Help Modal/Dialog
    if st.session_state.get('show_help', False):
        show_help_dialog()

    # Simple instructions
    st.markdown("""
    <div class="info-box">
        <h3>🏏 Welcome!</h3>
        <p>Find out which team is likely to win! Just fill in the details below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Teams
    st.subheader("⚔️ Which teams are playing?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏏 Team Chasing")
        st.caption("Batting now")
        batting_team = st.selectbox(
            'Team batting now',
            sorted(teams),
            key='batting',
            label_visibility='collapsed'
        )

    with col2:
        st.markdown("### ⚾ Team Defending")
        st.caption("Bowling now")
        bowling_team = st.selectbox(
            'Team bowling now',
            sorted(teams),
            key='bowling',
            label_visibility='collapsed'
        )

    with col3:
        st.markdown("### 🏟️ Stadium")
        st.caption("Match location")
        selected_city = st.selectbox(
            'Match city',
            sorted(cities),
            key='city',
            label_visibility='collapsed'
        )

    # Validation: Teams should be different
    if batting_team == bowling_team:
        st.error("⚠️ Please select different teams!")
        st.stop()

    st.markdown("---")

    # Step 2: Target
    st.subheader("🎯 What's the target score?")

    col_target1, col_target2 = st.columns([2, 1])

    with col_target1:
        target = st.number_input(
            'Runs needed to win',
            min_value=1,
            max_value=300,
            value=180,
            step=1,
            help=f"{batting_team} needs to score this many runs to win"
        )

    with col_target2:
        st.markdown(f"""
        <div class="scenario-box" style="margin-top: 0;">
            <h4>📊 Match Info</h4>
            <p><strong>{batting_team}</strong> needs <strong>{target}</strong> runs to win</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Step 3: Current Score
    st.subheader(f"📊 What's the current score?")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("### 📈 Runs")
        st.caption(f"{batting_team} has scored")
        score = st.number_input(
            'Runs scored',
            min_value=0,
            max_value=300,
            value=50,
            step=1,
            label_visibility='collapsed'
        )

    with col4:
        st.markdown("### ⏱️ Overs")
        st.caption("Overs played (e.g. 10.3)")
        overs = st.number_input(
            'Overs',
            min_value=0.1,
            max_value=19.5,
            value=10.0,
            step=0.1,
            format="%.1f",
            label_visibility='collapsed'
        )

    with col5:
        st.markdown("### 🚫 Wickets")
        st.caption("Players out")
        wickets = st.number_input(
            'Wickets',
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            label_visibility='collapsed'
        )

    # Match Summary
    st.markdown("---")
    runs_needed = target - score
    balls_left_calc = 120 - int(overs * 6)
    wickets_remaining = 10 - wickets

    st.markdown(f"""
    <div class="scenario-box">
        <h3>📺 Match Summary</h3>
        <h4>{batting_team} vs {bowling_team} at {selected_city}</h4>
        <p style="font-size: 1.1rem;">
            <strong>{batting_team}</strong> needs <strong>{runs_needed} more runs</strong> to win<br>
            Current Score: <strong>{score}/{wickets}</strong> in <strong>{overs}</strong> overs
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Predict button
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

    with predict_col2:
        predict_button = st.button('🔮 WHO WILL WIN?', use_container_width=True, type="primary")

    if predict_button:
        # Validation
        if score >= target:
            st.success(f"🎉 **{batting_team} has already WON the match!** They chased down the target!")
            st.balloons()
            st.stop()

        if wickets >= 10:
            st.error(f"🏁 **{batting_team} is ALL OUT!** {bowling_team} wins by {target - score - 1} runs!")
            st.stop()

        if overs >= 20:
            st.error(f"⏰ **Innings Over!** {bowling_team} wins by {target - score - 1} runs!")
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

        # NEW ENHANCED FEATURES
        pressure = rrr - crr
        runs_per_wicket = runs_left / wickets_left if wickets_left > 0 else runs_left * 10
        recent_runs = score  # Simplified for app
        run_rate_diff = crr - rrr

        # Match situation
        def get_situation():
            if wickets_left == 0:
                return 'impossible'
            rpo_required = runs_left / (balls_left / 6) if balls_left > 0 else 0
            if rpo_required <= 6:
                return 'easy'
            elif rpo_required <= 9:
                return 'moderate'
            elif rpo_required <= 12:
                return 'tough'
            else:
                return 'very_tough'

        situation = get_situation()

        # Create input dataframe with all features
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr],
            'pressure': [pressure],
            'runs_per_wicket': [runs_per_wicket],
            'recent_runs': [recent_runs],
            'run_rate_diff': [run_rate_diff],
            'situation': [situation]
        })

        # Make prediction
        try:
            result = model.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            # Display results - SIMPLE & USER FRIENDLY
            st.markdown("---")
            st.markdown("## 🏆 PREDICTION RESULT")

            # Main prediction display
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 2.5rem; border-radius: 15px; text-align: center; box-shadow: 0 10px 30px rgba(132, 250, 176, 0.4);'>
                    <h2 style='color: #1a365d; margin: 0;'>🏏 {batting_team}</h2>
                    <h1 style='color: #1a365d; font-size: 4rem; margin: 1rem 0;'>{round(win_prob * 100)}%</h1>
                    <p style='font-size: 1.2rem; color: #2c5282; font-weight: 600;'>Chance to Win</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(win_prob)

            with col_b:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 2.5rem; border-radius: 15px; text-align: center; box-shadow: 0 10px 30px rgba(250, 112, 154, 0.4);'>
                    <h2 style='color: #742a2a; margin: 0;'>⚾ {bowling_team}</h2>
                    <h1 style='color: #742a2a; font-size: 4rem; margin: 1rem 0;'>{round(loss_prob * 100)}%</h1>
                    <p style='font-size: 1.2rem; color: #9c4221; font-weight: 600;'>Chance to Win</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(loss_prob)

            st.markdown("---")

            # Simple interpretation
            if win_prob > 0.7:
                st.success(f"🟢 **{batting_team}** is likely to win this match!")
            elif win_prob > 0.5:
                st.info(f"🟡 **{batting_team}** has a slight advantage, but it's close!")
            elif win_prob > 0.3:
                st.warning(f"🟠 **{bowling_team}** is ahead right now!")
            else:
                st.error(f"🔴 **{bowling_team}** is likely to win this match!")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #000000; border-radius: 10px;'>
        <p style='font-size: 1.1rem;'>Made with ❤️ for Cricket Fans</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
