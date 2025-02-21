# -*- coding: utf-8 -*-
"""Insurence_pred.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QtMnoUU8BUf3gGUQZsQxDKiTASLay371
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

!gdown 1NBk1TFkK4NeKdodR2DxIdBp2Mk1mh4AS

df = pd.read_csv('insurance.csv')
print(df.head())

df.shape

df.info()

df.isnull().sum()



"""There are 986 rows and 11 columns No missing data"""

df.describe()

"""Average Age is 41.75 years, with Minimum age 18 years and maximum 66 years
Average height is 168.18 cm and minimum height 145 cm and maximum height 188 cm.
Average weight is 76.95 Kg with minimum weight 51 kg and maximum 132 kg
average premium price is 24336.71 with minimum 15000 and maximum 40000. All values looks acceptable and there are no impossible values in the data from description.
"""

# Distributuion of premium price
sns.histplot(df['PremiumPrice'], kde=True, color='blue')
plt.title('Premium Price Distribution')
plt.show()

import scipy.stats as stats
from scipy.stats import shapiro
# Step 1: Q-Q Plot and Shapiro-Wilk Test
plt.figure(figsize=(14, 6))

# Q-Q plot for original PremiumPrice
plt.subplot(1, 2, 1)
stats.probplot(df['PremiumPrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot (Original PremiumPrice)')

# Shapiro-Wilk Test for normality on original data
stat, p_value = shapiro(df['PremiumPrice'])
print("Shapiro-Wilk Test on Original Data (PremiumPrice):")
print(f"Statistic: {stat}, p-value: {p_value}")

# Calculate BMI
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

"""New feature BMI is created using height and weight"""

# Distributuion of age
sns.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution')
plt.show()

# Distributuion of age
sns.histplot(df['BMI'], kde=True, color='blue')
plt.title('BMI Distribution')
plt.show()

"""Age and BMI have near-normal distributions.
Premium price is right-skewed and doesn't fit a normal distribution well, even after transformation.

"""

sns.scatterplot(x='Age', y='PremiumPrice', hue='Diabetes', data=df)
plt.title('Premium Price vs Age (Colored by Diabetes)', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Premium Price', fontsize=12)
#plt.legend( labels=['No', 'Yes'])
plt.show()

sns.scatterplot(x='BMI', y='PremiumPrice', hue='BloodPressureProblems', data=df)
plt.title('Premium Price vs Age (Colored by Diabetes)', fontsize=14)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Premium Price', fontsize=12)
#plt.legend( labels=['No', 'Yes'])
plt.show()

"""It is observed in general high premium price for higher age but the relation is not linear but complex.
relation between BMI and premium price is also corelated observed from Tableau Dashboad but patern looks again complex
"""

# IQR Method for Outlier Detection
for column in ['Age', 'Height', 'Weight', 'PremiumPrice']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers")

# Boxplot for Outliers in Premium Price
sns.boxplot(df['PremiumPrice'])
plt.title('Outliers in Premium Price')
plt.show()

"""There are not much outliers exist, the outliers in Premium price are not seems to be unrealistic and not considering outlier sensitive model such as KNN decided to keep the outliers, I have checked by remving outlier performce of models didnt find much diffrence."""

#Box plot
columns_to_plot = [ "BloodPressureProblems", "AnyChronicDiseases", "KnownAllergies", "NumberOfMajorSurgeries"]


# Box plot primiumprie by Diabetes
for col in columns_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x= col, y='PremiumPrice', data=df)
    plt.title(f'Premium Price by {col}' )
    plt.show()

#Box plot
columns_to_plot = [ "AnyTransplants", "HistoryOfCancerInFamily"]


# Box plot primiumprie by Diabetes
for col in columns_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x= col, y='PremiumPrice', data=df)
    plt.title(f'Premium Price by {col}' )
    plt.show()

# Box plot primiumprie by AnyTransplants
sns.boxplot(x='AnyTransplants', y='PremiumPrice', data=df)
plt.title('Premium Price by AnyTransplants')
plt.show()

# Box plot primiumprie by AnyChronicDiseases
sns.boxplot(x='AnyChronicDiseases', y='PremiumPrice', data=df)
plt.title('Premium Price by AnyChronicDiseases')
plt.show()

# Box plot primiumprie by HistoryOfCancerInFamily
sns.boxplot(x='HistoryOfCancerInFamily', y='PremiumPrice', data=df)
plt.title('Premium Price by HistoryOfCancerInFamily')
plt.show()

"""The mean premium price for people with chronic diseases is higher than those without.
The mean premium price for diabetic individuals is higher than non-diabetics.
Individuals with blood pressure problems have a higher mean premium than those without.
No significant difference in mean premium price between individuals with and without allergies
Individuals with a family history of cancer have a higher mean premium than those without.
The mean premium price increases with the number of major surgeries.

Chronic Diseases vs. Premium Price
Null Hypothesis (H₀): The mean premium price for individuals with chronic diseases is equal to the mean premium price for those without chronic diseases.
Alternative Hypothesis (H₁): The mean premium price for individuals with chronic diseases is different from those without chronic diseases.
"""

from scipy.stats import ttest_ind
# T-test: Chronic Diseases vs. Premium Price
group1 = df[df['AnyChronicDiseases'] == 0]['PremiumPrice']
group2 = df[df['AnyChronicDiseases'] == 1]['PremiumPrice']
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")

"""The p-value is much smaller than the common significance level (0.05), so we reject the null hypothesis. This suggests that having chronic diseases significantly affects premium pricing.

H₀: There is no difference in mean premium price between individuals with and without diabetes.
H₁: There is a difference in mean premium price for individuals with and without diabetes.
"""

# T-test:Diabetes vs. Premium Price
group1 = df[df['Diabetes'] == 0]['PremiumPrice']
group2 = df[df['Diabetes'] == 1]['PremiumPrice']
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")

"""Since the p-value is less than 0.05, we reject the null hypothesis. This suggests that diabetes has a statistically significant impact on premium pricing.

H₀: No difference in mean premium price between individuals with and without blood pressure problems.
H₁: A significant difference in premium price between individuals with and without blood pressure problems.
"""

# T-test:BloodPressureProblems vs. Premium Price
group1 = df[df['BloodPressureProblems'] == 0]['PremiumPrice']
group2 = df[df['BloodPressureProblems'] == 1]['PremiumPrice']
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")

"""The p-value is very small, leading us to reject the null hypothesis. Blood pressure problems significantly affect premium pricing.

H₀: No difference in mean premium price between individuals with and without known allergies.
H₁: A significant difference in premium price based on allergies.
"""

# T-test:KnownAllergies vs. Premium Price
group1 = df[df['KnownAllergies'] == 0]['PremiumPrice']
group2 = df[df['KnownAllergies'] == 1]['PremiumPrice']
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")

"""Since the p-value is much greater than 0.05, we fail to reject the null hypothesis. This suggests that allergies do not significantly impact premium pricing.

H₀: No difference in mean premium price based on family history of cancer.
H₁: A significant difference in premium price based on family history of cancer.
"""

# T-test:HistoryOfCancerInFamily vs. Premium Price
group1 = df[df['HistoryOfCancerInFamily'] == 0]['PremiumPrice']
group2 = df[df['HistoryOfCancerInFamily'] == 1]['PremiumPrice']
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")

"""Null Hypothesis (H₀): There is no difference in mean premium price across different numbers of major surgeries.
Alternative Hypothesis (H₁): There is at least one significant difference in premium price among the different surgery groups.

The p-value is less than 0.05, so we reject the null hypothesis. A family history of cancer has a statistically significant impact on premium pricing.
"""

from scipy.stats import f_oneway
# Combine groups with 2 and 3 surgeries into a single group
df['NumberOfMajorSurgeries_Combined'] = df['NumberOfMajorSurgeries'].apply(lambda x: x if x < 2 else 2)

# Perform ANOVA on the combined groups
anova_results = f_oneway(
    df[df['NumberOfMajorSurgeries_Combined'] == 0]['PremiumPrice'],
    df[df['NumberOfMajorSurgeries_Combined'] == 1]['PremiumPrice'],
    df[df['NumberOfMajorSurgeries_Combined'] == 2]['PremiumPrice']
)

print(f"ANOVA: F-statistic = {anova_results.statistic}, p-value = {anova_results.pvalue}")

"""The p-value is extremely small, leading us to reject the null hypothesis. This indicates that the number of major surgeries significantly impacts premium pricing.

"""

from scipy.stats import spearmanr

# Perform Spearman correlation
corr, p_value = spearmanr(df['Age'], df['PremiumPrice'])

print(f"Spearman Correlation Coefficient: {corr}, p-value: {p_value}")

corr, p_value = spearmanr(df['BMI'], df['PremiumPrice'])

print(f"Spearman Correlation Coefficient: {corr}, p-value: {p_value}")

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Split data for model
X = df.drop(columns=['PremiumPrice','Weight' ,'Height','NumberOfMajorSurgeries_Combined' ])
y = df['PremiumPrice']

X_t = pd.DataFrame(X, columns=X.columns)
vif = pd.DataFrame()

vif['Features'] = X_t.columns
vif['VIF'] = [variance_inflation_factor(X_t.values, i) for i in range(X_t.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

from sklearn.model_selection import train_test_split
# Random data points are split.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardization
from sklearn.preprocessing import StandardScaler
numerical_features = ['Age', 'BMI', 'NumberOfMajorSurgeries']

std=StandardScaler()

X_train[numerical_features] = std.fit_transform(X_train[numerical_features])
X_test[numerical_features] = std.transform(X_test[numerical_features])

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Predict on training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Ensure all variables are numpy arrays
y_train = np.array(y_train, dtype=np.float64)
y_train_pred = np.array(y_train_pred, dtype=np.float64)
y_test = np.array(y_test, dtype=np.float64)
y_test_pred = np.array(y_test_pred, dtype=np.float64)

# Evaluate training data
print("Linear Regression (Training):")
print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_train, y_train_pred):.2f}")
#print(f"  Root Mean Squared Error (RMSE): {mean_squared_error(y_train, y_train_pred, squared=False):.2f}")
print(f"  R² Score: {r2_score(y_train, y_train_pred):.2f}")
print("-" * 50)

# Evaluate test data
print("Linear Regression (Test):")
print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_test_pred):.2f}")
#print(f"  Root Mean Squared Error (RMSE): {mean_squared_error(y_test, y_test_pred, squared=False):.2f}")
print(f"  R² Score: {r2_score(y_test, y_test_pred):.2f}")

"""Performed Linear Regression to establish a baseline model.
Evaluated performance using MAE (Mean Absolute Error) and R² Score.
Observed R² Score of 0.61 (Train) and 0.71 (Test), indicating the model explains 61%-71% of the variance in premium pricing.
"""

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=10)  # Try different alphas (1, 10, 100)
ridge.fit(X_train, y_train)
print("Ridge Test R² Score:", ridge.score(X_test, y_test))

lasso = Lasso(alpha=0.1)  # Can be adjusted for feature selection
lasso.fit(X_train, y_train)
print("Lasso Test R² Score:", lasso.score(X_test, y_test))

# Extract feature importance from Lasso model
lasso_coeffs = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': lasso.coef_})
lasso_coeffs = lasso_coeffs.sort_values(by='Coefficient', ascending=False)
print(lasso_coeffs)

def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name}:")
    print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}")
    #print(f"  Root Mean Squared Error (RMSE): {mean_squared_error(y_true, y_pred, squared=False):.2f}")
    print(f"  R² Score: {r2_score(y_true, y_pred):.2f}")
    print("-" * 50)
selected_features = ['AnyTransplants', 'Age', 'AnyChronicDiseases', 'HistoryOfCancerInFamily', 'BMI', 'KnownAllergies','NumberOfMajorSurgeries']

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Retrain Lasso
lasso_selected = Lasso(alpha=0.1)
lasso_selected.fit(X_train_selected, y_train)
evaluate_model(y_test, lasso_selected.predict(X_test_selected), "Lasso (Selected Features)")

"""Applied Ridge (L2) and Lasso (L1) regression to reduce overfitting and perform feature selection.
Lasso Regression provided feature importance by driving some coefficients to zero.
The top selected features: AnyTransplants, Age, AnyChronicDiseases, HistoryOfCancerInFamily, BMI.
"""

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate the model on training and test data
evaluate_model(y_train, rf_model.predict(X_train), "Random Forest (Training)")
evaluate_model(y_test, rf_model.predict(X_test), "Random Forest (Test)")

"""Used Random Forest (ensemble method) to capture non-linear relationships.
This is clearly showing overfitting
"""

rf = RandomForestRegressor(
    n_estimators=100,       # Reduce number of trees
    max_depth=10,           # Limit tree depth
    min_samples_split=5,    # Minimum samples to split a node
    min_samples_leaf=4,     # Minimum samples per leaf
    random_state=42
)

rf.fit(X_train, y_train)
# Evaluate the model on training and test data
evaluate_model(y_train, rf.predict(X_train), "Random Forest (Training)")
evaluate_model(y_test, rf.predict(X_test), "Random Forest (Test)")

"""Reduced complexity by tuning hyperparameters (max_depth, min_samples_split, etc.)"""

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
})

# Sort in descending order
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display feature importance
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()

# List of features to drop (low importance)
features_to_drop = ["KnownAllergies", "Diabetes", "BloodPressureProblems"]

# Remove these features from X_train and X_test
X_train_reduced = X_train.drop(columns=features_to_drop)
X_test_reduced = X_test.drop(columns=features_to_drop)

# Initialize and train Random Forest on the reduced dataset
rf_reduced = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)

rf_reduced.fit(X_train_reduced, y_train)

# Evaluate new model
evaluate_model(y_train, rf_reduced.predict(X_train_reduced), "Random Forest (Training) - Reduced Features")
evaluate_model(y_test, rf_reduced.predict(X_test_reduced), "Random Forest (Test) - Reduced Features")

# Get new feature importance
new_feature_importance = pd.DataFrame({
    'Feature': X_train_reduced.columns,
    'Importance': rf_reduced.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print new feature importance
print(new_feature_importance)

"""Identified Age, AnyTransplants, BMI, and Chronic Diseases as the most influential factors.
Dropped low-importance features (KnownAllergies, Diabetes, BloodPressureProblems).
"""

# Initialize and train the XGBoost model
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate on test data
evaluate_model(y_test, xgb_model.predict(X_test), "XGBoost (Test)")

y_pred_ls = lasso_selected.predict(X_test_selected)  # Lasso Regression (selected features)
y_pred_rf_reduced = rf_reduced.predict(X_test_reduced)  # Reduced Random Forest
y_pred_xgb = xgb_model.predict(X_test)  # Optimized XGBoost

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Scatter Plot for Random Forest (Original)
axs[0].scatter(y_test, y_pred_ls, alpha=0.5, color="blue")
axs[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="black")
axs[0].set_title("Linear regression")
axs[0].set_xlabel("Actual Values (y_test)")
axs[0].set_ylabel("Predicted Values (y_pred)")

# Scatter Plot for Reduced Random Forest
axs[1].scatter(y_test, y_pred_rf_reduced, alpha=0.5, color="green")
axs[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="black")
axs[1].set_title("Random Forest (Reduced)")
axs[1].set_xlabel("Actual Values (y_test)")
axs[1].set_ylabel("Predicted Values (y_pred)")

# Scatter Plot for XGBoost
axs[2].scatter(y_test, y_pred_xgb, alpha=0.5, color="red")
axs[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="black")
axs[2].set_title("XGBoost")
axs[2].set_xlabel("Actual Values (y_test)")
axs[2].set_ylabel("Predicted Values (y_pred)")

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plots
plt.show()

"""Fine-tune alpha (regularization strength) in Lasso and Ridge models using cross-validation can be done for linear regression model.
In case of random forest further optimize hyperparameters using GridSearchCV can be done, also experiment with non-linear feature interactions for better accuracy.for XGboost hyperparameter tuning using GridSearchCV can be perform.


"""

# saving the model
import pickle

pickle_out = open("insurancenw.pkl", mode = "wb")
pickle.dump(rf_reduced, pickle_out)
pickle_out.close()

import joblib

# Load the trained model
model = joblib.load('insurancenw.pkl')

# Example input data for prediction (new input to predict premium)
input_data = {
    'Age': 20,


    'AnyTransplants': 0,  # No
    'AnyChronicDiseases': 0,  # Yes

    'HistoryOfCancerInFamily': 0,  # Yes
    'NumberOfMajorSurgeries': 0,
    'BMI': 30.0
}

# Convert the input data into a DataFrame (same format as your training data)
input_df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(input_df)

# Output the prediction
print(f"Estimated Insurance Premium: ${prediction[0]:.2f}")

"""# New Section"""