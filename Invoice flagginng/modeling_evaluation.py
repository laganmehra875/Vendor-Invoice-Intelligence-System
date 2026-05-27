from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Classifier.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier using GridSearchCV.
    """
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 4, 6],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ['gini', 'entropy']
    }
    
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a classification model and return metrics.
    """
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
