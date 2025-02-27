
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Create the DataFrame
data = pd.DataFrame({
    'Age': [22, 25, 30, 35],
    'Income': [50000, 60000, 55000, 45000],
    'Buy': ['No', 'Yes', 'Yes', 'No']
})

# Encode the target variable (Yes/No) into binary
le = LabelEncoder()
data['Buy'] = le.fit_transform(data['Buy'])

# Split into features and target
X = data[['Age', 'Income']]
y = data['Buy']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI part
st.title('Customer Purchase Prediction')

# Inputs from user
age = st.number_input('Enter Age', min_value=18, max_value=100)
income = st.number_input('Enter Income', min_value=10000, max_value=200000)

if st.button('Predict'):
    prediction = model.predict([[age, income]])
    if prediction == 1:
        st.write("Prediction: The customer is likely to buy.")
    else:
        st.write("Prediction: The customer is unlikely to buy.")
