import mlflow
import pandas as pd
import json
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Model and Artifacts at Startup ---

# IMPORTANT: Replace this with the actual Run ID of your trained model
LOGGED_MODEL_RUN_ID = '6a03bfee78f44b3e8eb9f93df847995b' # <-- PASTE YOUR RUN ID HERE

if LOGGED_MODEL_RUN_ID == 'YOUR_RUN_ID_HERE':
    print("="*50)
    print("ERROR: Please update the 'LOGGED_MODEL_RUN_ID' in app.py")
    print("You can find the Run ID in the MLflow UI.")
    print("="*50)
    exit()

# Load the trained pipeline model from MLflow
logged_model_uri = f"runs:/{LOGGED_MODEL_RUN_ID}/player-predictor-model"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Load the label encoder classes
# Note: MLflow automatically downloads artifacts to a temp directory.
# We need to construct the path to the downloaded artifact.
client = mlflow.tracking.MlflowClient()
local_path = client.download_artifacts(LOGGED_MODEL_RUN_ID, "model_meta/label_encoder_classes.json")

le = LabelEncoder()
with open(local_path, 'r') as f:
    # The LabelEncoder expects a NumPy array for its classes.
    le.classes_ = np.array(json.load(f))

print("Model and Label Encoder loaded successfully.")

# --- Load data for dropdowns ---
try:
    df = pd.read_csv("players_dataset.csv")
    unique_clubs = sorted(df['club'].unique())
    unique_national_teams = sorted(df['national_team'].unique())
    print("Dropdown data for UI loaded successfully.")
except FileNotFoundError:
    print("ERROR: players_dataset.csv not found. UI dropdowns will be empty.")
    unique_clubs = []
    unique_national_teams = []

# --- 2. Create Flask Application ---
app = Flask(__name__)

# Define the main route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        # Get data from the form
        user_club = request.form.get('club')
        user_national_team = request.form.get('national_team')
        user_age = int(request.form.get('age'))

        # Create a DataFrame for prediction
        new_user_data = pd.DataFrame({
            'club': [user_club],
            'national_team': [user_national_team],
            'age': [user_age]
        })

        # Predict using the loaded model
        predicted_label_encoded = loaded_model.predict(new_user_data)
        prediction_result = le.inverse_transform(predicted_label_encoded)[0]

    # Render the HTML page, passing the prediction result if it exists
    return render_template(
        'index.html',
        prediction=prediction_result,
        clubs=unique_clubs,
        national_teams=unique_national_teams
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on a different port than mlflow ui