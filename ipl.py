import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# --------------------------
# 1. Load dataset
# --------------------------
df = pd.read_csv('D:\\projects\\sid\\siddd.venv\\ipl.csv')

# --------------------------
# 2. Remove unwanted columns
# --------------------------
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
print('Before removing unwanted columns: {}'.format(df.shape))
df.drop(labels=columns_to_remove, axis=1, inplace=True)
print('After removing unwanted columns: {}'.format(df.shape))

# --------------------------
# 3. Keep only consistent teams
# --------------------------
consistent_teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]

print('Before removing inconsistent teams: {}'.format(df.shape))
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(df.shape))

# --------------------------
# 4. Remove first 5 overs data
# --------------------------
print('Before removing first 5 overs data: {}'.format(df.shape))
df = df[df['overs'] >= 5.0]
print('After removing first 5 overs data: {}'.format(df.shape))

# --------------------------
# 5. Convert 'date' column to datetime
# --------------------------
print("Before converting 'date':", type(df.iloc[0, 0]))
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
print("After converting 'date':", type(df.iloc[0, 0]))

# --------------------------
# 6. Correlation heatmap
# --------------------------
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(13, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn')
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()

# --------------------------
# 7. One-hot encode teams
# --------------------------
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# --------------------------
# 8. Keep required columns
# --------------------------
encoded_df = encoded_df[['date', 
    'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
    'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
    'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
    'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
    'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
    'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
    'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# --------------------------
# 9. Train-Test split
# --------------------------
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Remove 'date' column
X_train.drop(labels='date', axis=1, inplace=True)
X_test.drop(labels='date', axis=1, inplace=True)

print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))

# --------------------------
# 10. Train Models
# --------------------------
# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_lr = linear_regressor.predict(X_test)

print("---- Linear Regression - Model Evaluation ----")
print("MAE: {}".format(mae(y_test, y_pred_lr)))
print("MSE: {}".format(mse(y_test, y_pred_lr)))
print("RMSE: {}".format(np.sqrt(mse(y_test, y_pred_lr))))

# Decision Tree Regression
decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train, y_train)
y_pred_dt = decision_regressor.predict(X_test)

print("---- Decision Tree Regression - Model Evaluation ----")
print("MAE: {}".format(mae(y_test, y_pred_dt)))
print("MSE: {}".format(mse(y_test, y_pred_dt)))
print("RMSE: {}".format(np.sqrt(mse(y_test, y_pred_dt))))

# Random Forest Regression
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train, y_train)
y_pred_rf = random_regressor.predict(X_test)

print("---- Random Forest Regression - Model Evaluation ----")
print("MAE: {}".format(mae(y_test, y_pred_rf)))
print("MSE: {}".format(mse(y_test, y_pred_rf)))
print("RMSE: {}".format(np.sqrt(mse(y_test, y_pred_rf))))

# AdaBoost Regression
adb_regressor = AdaBoostRegressor(estimator=linear_regressor, n_estimators=100)
adb_regressor.fit(X_train, y_train)
y_pred_adb = adb_regressor.predict(X_test)

print("---- AdaBoost Regression - Model Evaluation ----")
print("MAE: {}".format(mae(y_test, y_pred_adb)))
print("MSE: {}".format(mse(y_test, y_pred_adb)))
print("RMSE: {}".format(np.sqrt(mse(y_test, y_pred_adb))))

# --------------------------
# 11. Prediction Function
# --------------------------
def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians',
                  overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
    temp_array = []

    # Batting Team
    bat_teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 
                 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 
                 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    for team in bat_teams:
        temp_array.append(1 if batting_team == team else 0)

    # Bowling Team
    for team in bat_teams:
        temp_array.append(1 if bowling_team == team else 0)

    # Match stats
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])

    temp_array = np.array([temp_array])

    # Prediction
    predicted_score = int(linear_regressor.predict(temp_array)[0])
    return predicted_score

# --------------------------
# 12. Test Prediction
# --------------------------
final_score = predict_score(batting_team='Kolkata Knight Riders', bowling_team='Delhi Daredevils',
                             overs=9.2, runs=79, wickets=2, runs_in_prev_5=60, wickets_in_prev_5=1)

print("The final predicted score (range): {} to {}".format(final_score-10, final_score+5))
