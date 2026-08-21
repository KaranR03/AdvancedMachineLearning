import joblib

def load_best_model():
    best_model = joblib.load('best_model.joblib')
    return best_model

def predict_quality(X_normed_features):
    best_model = load_best_model()
    return best_model.predict(X_normed_features)