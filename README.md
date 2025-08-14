# ipl-score-prediction
Machine learning models for predicting IPL cricket match scores using historical ball-by-ball data.
# 🏏 IPL Score Prediction

A machine learning project that predicts **final cricket match scores** in the Indian Premier League (IPL) using historical ball-by-ball data.  
This project applies **data preprocessing, feature engineering, and multiple regression models** to forecast match outcomes.

---

## 📌 Project Overview
This project:
- Cleans and preprocesses raw IPL match data
- Encodes categorical features such as batting and bowling teams
- Trains and evaluates multiple regression models:
  - **Linear Regression**
  - **Decision Tree Regression**
  - **Random Forest Regression**
  - **AdaBoost Regression**
- Provides a **custom prediction function** to estimate final match scores based on current match conditions

---

## 📊 Dataset
- **Source:** IPL ball-by-ball dataset (`ipl.csv`)
- **Columns used:**
  - `bat_team`, `bowl_team`
  - `overs`, `runs`, `wickets`
  - `runs_last_5`, `wickets_last_5`
  - `total` (final score — target variable)

> **Note:** The raw dataset is large and may be excluded from the repository. You can download it from Kaggle or other cricket statistics sources.

---

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/ipl-score-prediction.git
cd ipl-score-prediction
