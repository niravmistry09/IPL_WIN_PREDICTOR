# IPL_WIN_PREDICTOR
“End-to-end IPL match win prediction machine learning project with EDA and Streamlit app.”


🏏 IPL Win Probability Predictor  
A real-time IPL **win probability prediction** project built using Python, Machine Learning, and Streamlit.  
This model predicts the probability of a team winning **while the match is still going on**, based on live match conditions.



📊 Project Overview  
This end-to-end ML project covers:

- Complete **EDA** on ball-by-ball and match datasets  
- Feature engineering for live-match conditions (overs, runs, wickets, run rate, required rate)  
- Machine Learning model to estimate **win probability**  
- Interactive **Streamlit Web App** to test the model during a match  
- All code built inside a single Google Colab notebook



📁 Dataset Used  
This project uses 2 official IPL datasets:

1️⃣ matches.csv
Contains match-level information:
- Teams  
- Toss decision  
- Winner  
- Venue  
- Season  
- Margin of victory  

2️⃣ deliveries.csv
Contains ball-by-ball data:
- Batting team  
- Bowling team  
- Runs scored  
- Wickets  
- Overs/balls  
- Player dismissals
  Note: Dataset available on Kaggle (file size >25 MB, so not uploaded here).


🧠 Machine Learning Model  
Algorithm used: RandomForest Classifier
Target variable: Probability of chasing team winning the match

📌 Features used:
- Current score  
- Wickets fallen  
- Overs bowled  
- Current run rate (CRR)  
- Required run rate (RRR)  
- Runs required  
- Balls remaining  
- Match venue and teams  
 

