import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

DAGSHUB_USERNAME = "Rezka08"
DAGSHUB_REPO = "Sistem-ML-Rezka"
DAGSHUB_TOKEN = "d8a5eb07620097b107243d1fad5c25460a52b523"

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

def train_simple_model():
    # Load dataset hasil preprocessing Tahap 1
    # Pastikan path ini sesuai dengan letak dataset di dalam folder MLProject nanti
    df = pd.read_csv("titanic_preprocessing/dataset_ready.csv")
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Titanic_CI_Workflow")
    
    with mlflow.start_run(run_name="CI_Automated_Run"):
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"Model berhasil dilatih via CI Workflow! Akurasi: {acc:.2f}")

if __name__ == "__main__":
    train_simple_model()