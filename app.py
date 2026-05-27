import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("data/Titanic.csv")

# Load Trained Model
model = joblib.load("models/random_forest_model.pkl")

# Data Preprocessing
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Encode Categorical Columns
data["Sex"] = data["Sex"].map({
    "male": 1,
    "female": 0
})

data["Embarked"] = data["Embarked"].map({
    "C": 0,
    "Q": 1,
    "S": 2
})

# Features and Target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Accuracy Calculation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Streamlit Title
st.title("🚢 Titanic Survival Prediction")

# Display Accuracy
st.subheader("Model Accuracy")

st.success(f"Accuracy : {accuracy:.2f}")

# User Inputs
st.subheader("Enter Passenger Details")

Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

Age = st.slider(
    "Age",
    1,
    80,
    25
)

SibSp = st.number_input(
    "Siblings / Spouses Aboard",
    0,
    10,
    0
)

Parch = st.number_input(
    "Parents / Children Aboard",
    0,
    10,
    0
)

Fare = st.number_input(
    "Fare",
    0.0,
    600.0,
    50.0
)

Embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Encode Inputs
Sex = 1 if Sex == "male" else 0

embarked_map = {
    "C": 0,
    "Q": 1,
    "S": 2
}

Embarked = embarked_map[Embarked]

# Input DataFrame
input_data = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked": [Embarked]
})

# Prediction
if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
