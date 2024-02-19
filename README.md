# breasty-cancer-diagnosis
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# breast cancer load cheyanam
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# data ne training and testing set ilek aakuka
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression modeline create cheyth train cheyuka
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)

# testing set predict cheyuka
y_pred = logistic_reg.predict(X_test)

#model evaluate cheyuka
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_rep)

# confusion matrix ine visualise cheyukha
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
