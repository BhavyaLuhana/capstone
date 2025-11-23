import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
with open("svm_knn_data.pickle", "rb") as f:
    data = pickle.load(f)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)

print(f"KNN Accuracy: {acc * 100:.2f}%")

# Save model
joblib.dump(model, "knn_model.pkl")
print("KNN model saved as knn_model.pkl")