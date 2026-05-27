import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("data/titanic.csv")

# Drop Unnecessary Columns
drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]

for col in drop_columns:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Handle Missing Values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Encode Categorical Columns
label_encoder = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Target Column
target = "Survived"

# Features and Target
X = data.drop(target, axis=1)
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Save Model inside models folder
joblib.dump(model, "models/random_forest_model.pkl")

print("Model saved inside models folder!")