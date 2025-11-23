import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load flattened dataset
with open("svm_knn_data.pickle", "rb") as f:
    data = pickle.load(f)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# Create and train SVM model
model = SVC()
model.fit(x_train, y_train)

# Evaluate
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)

print(f"SVM Accuracy: {acc * 100:.2f}%")

# Save model
import joblib
joblib.dump(model, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")
