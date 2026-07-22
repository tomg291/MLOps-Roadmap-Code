import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def main():
    # Load in the data
    print("Loading data...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Hyperparameters
    n_estimators = 50
    max_depth = 2

    with mlflow.start_run():
        # Train a random forest 
        print("Training model...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.2f}")

        # Experiment tracking with MLFlow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random-forest-model")
        print("Experiment logged with MLFlow")

        # Generate confusion matrix 
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")

        # Write markdown file
        with open("metrics.md", "w") as outfile:
            outfile.write("## Model Evaluation\n")
            outfile.write(f"**Accuracy:** {accuracy:.2f}\n\n")
            outfile.write("### Confusion Matrix\n")
            outfile.write("![Confusion Matrix](./confusion_matrix.png)\n")

if __name__ == "__main__":
    main()