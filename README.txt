================================================================================
PREDICTING MLB GAME OUTCOMES: HOW GOOD ARE SPORTSBOOK MODELS?
Team 139 | Nathan Corsetti, Kenneth Corsetti, Aditya Bansal, Chirag Kulkarni
Georgia Tech CSE 6242 | Data and Visual Analytics
================================================================================

DESCRIPTION
-----------
This package contains an interactive Streamlit dashboard for predicting MLB game
outcomes and evaluating the efficiency of sportsbook implied probabilities. The
dashboard trains four machine learning models (Logistic Regression, Random Forest,
XGBoost, and CatBoost) on 23,240 MLB games from 2012-2021 and compares their
predicted probabilities against sportsbook implied probabilities.

The dashboard includes five interactive tabs:
  - Game Explorer: Browse all 2020-2021 test set games with model vs. sportsbook
    probabilities and gap values, filterable by team and season
  - Calibration: Interactive calibration curves comparing all models against the
    sportsbook benchmark with ECE values
  - Backtest: Simulated flat-stake betting ROI by gap threshold, adjustable via
    sidebar slider
  - Model Comparison: Side-by-side metrics table (Accuracy, AUC, Brier, ECE)
    with AUC and ECE bar charts
  - Gap Analysis: Probability gap distribution and actual win rate by gap quintile

Key finding: Logistic Regression achieves AUC 0.6086, nearly matching the
sportsbook benchmark of 0.6073, using only 7 engineered features.


INSTALLATION
------------
1. Ensure Python 3.8+ is installed on your system

2. Install required dependencies:
      pip install streamlit plotly scikit-learn xgboost catboost pandas numpy

3. Download the dataset from Kaggle:
      https://www.kaggle.com/datasets/tobycrabtree/mlb-scores-and-betting-data
      File needed: oddsDataMLB.csv

4. Place oddsDataMLB.csv in the same directory as app.py


EXECUTION
---------
From the directory containing app.py and oddsDataMLB.csv, run:

      streamlit run app.py

The dashboard will open automatically in your browser at http://localhost:8501

On first load, all four models will train and cache automatically (takes ~1-2
minutes). Subsequent loads will be instant due to Streamlit's caching.

Use the sidebar to:
  - Filter games by team
  - Filter by season (2020 or 2021)
  - Select the primary model to display
  - Adjust the betting gap threshold for the backtest simulation


A live version of the dashboard is deployed at:
      https://mlb-project-hdem9fkntmevjxjngx7qxy.streamlit.app/

================================================================================
