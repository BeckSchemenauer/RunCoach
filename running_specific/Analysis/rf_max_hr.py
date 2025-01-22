import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Function to convert mm:ss to seconds
def convert_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

# Load the dataset
file_path = "../../Data/run_data.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Drop rows with any empty (NaN) entries
data = data.dropna()

# Convert 'moving_time' column from mm:ss format to seconds
data['moving_time'] = data['moving_time'].apply(convert_to_seconds)
data['speed_distance_effect'] = data['mph'] ** 3 * data['moving_time']

# Feature columns
features = ["moving_time", "distance_km", "pace_per_km", "elevation_gain_m", "speed_distance_effect"]

# Create binary categories for max heart rate (below 175 and above or equal to 175)
target_max_hr = "max_heart_rate"  # Assuming you have a column for max heart rate
data[target_max_hr] = (data[target_max_hr] >= 180).astype(int)  # 0 for below 180, 1 for 180 and above

# Prepare data for max heart rate category prediction
X_max_hr = data[features]
y_max_hr = data[target_max_hr]

# Split into training and testing sets
X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(X_max_hr, y_max_hr, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_max, y_train_max)

# Define the classifiers with class weights to favor the minority class
gb_max_hr = GradientBoostingClassifier(random_state=42)
rf_max_hr = RandomForestClassifier(random_state=42, class_weight='balanced')  # Apply class weights
lr_max_hr = LogisticRegression(random_state=42, class_weight='balanced')  # Apply class weights
svc_max_hr = SVC(probability=True, random_state=42)  # Support Vector Classifier
xgb_max_hr = XGBClassifier(random_state=42)  # XGBoost Classifier

# Create the Voting Classifier
voting_clf_max = VotingClassifier(estimators=[
    ('gb', gb_max_hr),
    ('rf', rf_max_hr),
    ('lr', lr_max_hr),
    ('svc', svc_max_hr),
    ('xgb', xgb_max_hr)
], voting='soft')  # 'soft' voting uses predicted probabilities

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    # Gradient Boosting parameters
    'gb__n_estimators': [150],  # Number of boosting stages for GB
    'gb__learning_rate': [.03],  # Learning rate for GB
    'gb__max_depth': [7,],  # Maximum depth for GB
    'gb__max_features': ['sqrt'],  # Number of features to consider at each split for GB
    'gb__random_state': [4],

    # Random Forest parameters
    'rf__n_estimators': [125,],  # Number of trees for RF
    'rf__max_depth': [7,],  # Max depth for RF
    'rf__bootstrap': [True],
    'rf__random_state': [3],

    # Logistic Regression parameters
    'lr__C': [1,],  # Regularization strength for LR

    # Support Vector Classifier parameters
    'svc__C': [1],  # Regularization parameter for SVC
    'svc__kernel': ['rbf'],  # Kernel types for SVC

    # XGBoost parameters
    'xgb__n_estimators': [100,],  # Number of estimators for XGBoost
    'xgb__max_depth': [3],  # Maximum depth for XGBoost trees
    'xgb__learning_rate': [0.01],  # Learning rate for XGBoost
    'xgb__random_state': [0],
}

# Perform Grid Search with the VotingClassifier
grid_search_max = GridSearchCV(estimator=voting_clf_max, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_max.fit(X_train_balanced, y_train_balanced)

# Get the best model from grid search
best_voting_clf_max = grid_search_max.best_estimator_

# Print the best hyperparameters
print(f"Best Hyperparameters: {grid_search_max.best_params_}")

# Predict and evaluate for max heart rate categories using the best model
y_pred_max = best_voting_clf_max.predict(X_test_max)

# Print evaluation metrics
print("Max Heart Rate Category Prediction Metrics:")
print(f"Accuracy: {accuracy_score(y_test_max, y_pred_max):.2f}")
print("Classification Report:")
print(classification_report(y_test_max, y_pred_max))
