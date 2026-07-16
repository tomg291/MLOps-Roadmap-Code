from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load in the data
    print("Loading data...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Train a random forest 
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the model locally
    print("Saving model to model.joblib...")
    joblib.dump(model, 'model.joblib')
    print("Done!")

if __name__ == "__main__":
    main()