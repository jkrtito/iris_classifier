from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
# Load dataset
iris =load_iris()
x =iris.data # shape (150,4)
y =iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

# split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
#predict
y_pred = model.predict(x_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])   

#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 

#confusion matrix
cm = confusion_matrix (y_test, y_pred)

os.makedirs("outputs", exist_ok=True)

