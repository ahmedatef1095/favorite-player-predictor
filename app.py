import mlflow
import pandas as pd
import json
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Latest Model from MLflow Model Registry ---

# Load the latest version of the registered model.
# This is much more robust than using a hardcoded run_id.
model_name = "player-predictor"
logged_model_uri = f"models:/{model_name}/latest"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Load the label encoder classes associated with the loaded model
client = mlflow.tracking.MlflowClient()
run_id_of_loaded_model = loaded_model.metadata.run_id
local_path = client.download_artifacts(run_id_of_loaded_model, "model_meta/label_encoder_classes.json")

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
    app.run(debug=True, port=5001) # Run on a different port than mlflow ui