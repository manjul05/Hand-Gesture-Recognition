import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("[INFO] Training model...")

with open("data.pkl", "rb") as f:
    data, labels = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Model saved to 'gesture_model.pkl'")
