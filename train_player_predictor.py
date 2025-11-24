import mlflow
import mlflow.sklearn
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# It's good practice to name your experiments
mlflow.set_experiment("Favorite Player Predictor")

# --- 1. Data Preparation ---
# Load the dataset from the CSV file
df = pd.read_csv("players_dataset.csv")
X = df[['club', 'national_team', 'age']]
y = df['favorite_player']

# Encode the target variable (player names) into numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# --- 2. Model Training with Preprocessing ---
# Create a preprocessor to handle categorical features ('club', 'national_team')
# OneHotEncoder converts them to numeric format. 'passthrough' leaves the 'age' column as is.
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['club', 'national_team']),
    remainder='passthrough'
)

# Define the model parameters
params = {
    "max_depth": 5,
    "min_samples_leaf": 1,
    "random_state": 42
}

# Create a scikit-learn pipeline
# This pipeline first runs the preprocessor and then trains the classifier.
model_pipeline = make_pipeline(
    preprocessor,
    DecisionTreeClassifier(**params)
)

# --- 3. MLflow Tracking ---
with mlflow.start_run() as run:
    print("Starting MLflow run...")

    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.2f}")

    # Log parameters, metrics, and the model
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the entire pipeline (preprocessor + model)
    # This is crucial for making predictions on new, raw data later
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="player-predictor-model",
        # Add an input example to automatically log the model signature
        input_example=X_train.head(1)
    )
    
    # Save the label encoder classes to a file first
    label_encoder_path = "label_encoder_classes.json"
    with open(label_encoder_path, 'w') as f:
        json.dump(le.classes_.tolist(), f)

    # Log the file as an artifact. The second argument specifies the destination path
    # within the MLflow run's artifact store.
    mlflow.log_artifact(label_encoder_path, artifact_path="model_meta")

    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print("Model, parameters, and metrics have been logged.")


# --- 4. Example of Loading and Predicting ---
print("\n--- Example Prediction (from training script) ---")
# This shows how you would use the logged model in a different application
logged_model_uri = f"runs:/{run_id}/player-predictor-model"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Create new data for a user
new_user_data = pd.DataFrame({
    'club': ['Liverpool'],
    'national_team': ['Egypt'],
    'age': [32]
})

# Predict the encoded label
predicted_label_encoded = loaded_model.predict(new_user_data)

# Convert the numeric prediction back to the player's name
predicted_player = le.inverse_transform(predicted_label_encoded)

print(f"User Input: Club='{new_user_data['club'].iloc[0]}', National Team='{new_user_data['national_team'].iloc[0]}', Age={new_user_data['age'].iloc[0]}")
print(f"Predicted Favorite Player: {predicted_player[0]}")
