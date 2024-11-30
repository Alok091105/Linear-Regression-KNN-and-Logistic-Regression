# Linear-Regression-KNN-and-Logistic-Regression

#LINEAR REGRESSION 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv('Food_and_Nutrition__.csv')


data['Activity Level'] = data['Activity Level'].map({
    'Sedentary': 1, 'Lightly Active': 2, 'Moderately Active': 3, 'Very Active': 4
})

data = pd.get_dummies(data, columns=['Dietary Preference'], drop_first=True)

X = data[['Ages', 'Height', 'Weight', 'Activity Level']]  
y = data['Daily Calorie Target']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

new_data = [[35, 170, 75, 3]] 
predicted_calories = model.predict(new_data)
print("Predicted Daily Calories:", predicted_calories[0])



#LOGISTIC

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

data = pd.read_csv('Food_and_Nutrition__.csv')


data['Activity Level'] = data['Activity Level'].map({
    'Sedentary': 1, 'Lightly Active': 2, 'Moderately Active': 3, 'Very Active': 4
})

data = pd.get_dummies(data, columns=['Dietary Preference'], drop_first=True)


def categorize_calories(calories):
    if calories < 2000:
        return "low"
    elif 2000 <= calories <= 2500:
        return "medium"
    else:
        return "high"

data['Calorie Category'] = data['Daily Calorie Target'].apply(categorize_calories)

category_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['Calorie Category'] = data['Calorie Category'].map(category_mapping)

X = data[['Ages', 'Height', 'Weight', 'Activity Level']]  # Add more features if needed
y = data['Calorie Category']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_data = [[35, 170, 75, 3]] 
predicted_category = log_model.predict(new_data)
category_label = [key for key, value in category_mapping.items() if value == predicted_category[0]]
print("Predicted Calorie Category:", category_label[0])


#KNN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Food_and_Nutrition__.csv')


data['Activity Level'] = data['Activity Level'].map({
    'Sedentary': 1, 'Lightly Active': 2, 'Moderately Active': 3, 'Very Active': 4
})

data = pd.get_dummies(data, columns=['Dietary Preference'], drop_first=True)


X = data[['Ages', 'Height', 'Weight', 'Activity Level']]  
y = data['Daily Calorie Target']


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5) 
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


new_data = [[35, 170, 75, 3]]  
predicted_calories = knn_model.predict(new_data)
print("Predicted Daily Calories:", predicted_calories[0])









