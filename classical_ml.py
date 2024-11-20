import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Load extracted features and labels
all_features = torch.load('features/extracted_feat.pth')
all_labels = torch.load('features/labels.pth')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, 
	all_labels, test_size=0.2, random_state=42)


# # Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# rf_predictions = rf_classifier.predict(X_test)
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# print("Random Forest Accuracy:", rf_accuracy)
# print("Random Forest Classification Report:")
# print(classification_report(y_test, rf_predictions))

# # SVM Classifier
# svm_classifier = LinearSVC(loss='hinge', C=1.0, penalty='l2', max_iter=1000, random_state=42)
# svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_predictions)
# print("SVM Accuracy:", svm_accuracy)
# print("SVM Classification Report:")
# print(classification_report(y_test, svm_predictions))

# Decision Tree Classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)
# dt_classifier.fit(X_train, y_train)
# dt_predictions = dt_classifier.predict(X_test)
# dt_accuracy = accuracy_score(y_test, dt_predictions)
# print("Decision Tree Accuracy:", dt_accuracy)
# print("Decision Tree Classification Report:")
# print(classification_report(y_test, dt_predictions))


# # k-NN Classifier
# knn_classifier = KNeighborsClassifier(n_neighbors=5,  weights='distance', p=2)
# knn_classifier.fit(X_train, y_train)
# knn_predictions = knn_classifier.predict(X_test)

# knn_accuracy = accuracy_score(y_test, knn_predictions)
# print("k-NN Accuracy:", knn_accuracy)
# print("k-NN Classification Report:")
# print(classification_report(y_test, knn_predictions))


# # Create an XGBoost classifier multi: softprob
# xgb_classifier = XGBClassifier(n_estimators=300, max_depth=6, 
# 	objective='multi:softprob', random_state=42)
# xgb_classifier.fit(X_train, y_train)
# xgb_predictions = xgb_classifier.predict(X_test)

# # Calculate accuracy and classification report
# xgb_accuracy = accuracy_score(y_test, xgb_predictions)
# print("XGBoost Accuracy:", xgb_accuracy)
# print("XGBoost Classification Report:")
# print(classification_report(y_test, xgb_predictions))


# # Create an MLP classifier
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(512,), activation='tanh', 
# 	max_iter=200, random_state=42)
# mlp_classifier.fit(X_train, y_train)
# mlp_predictions = mlp_classifier.predict(X_test)

# # Calculate accuracy and classification report
# mlp_accuracy = accuracy_score(y_test, mlp_predictions)
# print("MLP Accuracy:", mlp_accuracy)
# print("MLP Classification Report:")
# print(classification_report(y_test, mlp_predictions))



# # Define a function to plot confusion matrix
# def plot_confusion_matrix(y_true, y_pred, title):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(title)

# plot_confusion_matrix(y_test, svm_predictions, "SVM Confusion Matrix")
# plt.show()

# SVM Classifier
svm_classifier = SVC(kernel='linear', C=1.0, max_iter=1000, 
	probability=True,random_state=42)
svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)

# Compute SVM ROC curve and ROC-AUC score
y_scores = svm_classifier.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for class_idx in range(62):
    fpr[class_idx], tpr[class_idx], _ = roc_curve((y_test == class_idx).astype(int), y_scores[:, class_idx])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Compute micro-average ROC curve and ROC-AUC score
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot SVM ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=2,
         label=f'SVM (AUC = {roc_auc["micro"]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC-AUC Curve')
plt.legend(loc="lower right")
plt.show()


"""
# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=750, criterion='gini', 
    	max_depth=50, random_state=42),
    "SVM": LinearSVC(loss='hinge', C=1.0, max_iter=1000, random_state=42),
	"MLP": MLPClassifier(hidden_layer_sizes=(512,), activation='tanh', 
		max_iter=200, random_state=42)


}

# Plot ROC-AUC curves for each classifier
plt.figure(figsize=(10, 8))
for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_scores = classifier.predict_proba(X_test)
    
    # Compute ROC curve and ROC-AUC score for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for class_idx in range(62):
        fpr[class_idx], tpr[class_idx], _ = roc_curve((y_test == class_idx).astype(int), y_scores[:, class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
    
    # Compute micro-average ROC curve and ROC-AUC score
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curve for each class
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{classifier_name} (AUC = {roc_auc["micro"]:.2f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curves')
plt.legend(loc="lower right")
plt.show()
"""