import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

def train_models(X_train, y_train):
    trained = {}

    # Logistic Regression: tuning 'C'
    for c in [0.1, 1.0, 10.0]:
        name = f"logistic_regression_C_{c}"
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(X_train, y_train)
        trained[name] = model

    # Random Forest: tuning 'max_depth'
    for depth in [5, 10, 15]:
        name = f"random_forest_depth_{depth}"
        model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        trained[name] = model

    # Naive Bayes (no tuning for now)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    trained["naive_bayes"] = nb_model

    return trained

def save_models(trained_models, output_dir):
    for name, model in trained_models.items():
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
