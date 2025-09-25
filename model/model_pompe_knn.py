import numpy as np
from creat_training_dataset import creat_training_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

data = creat_training_dataset("data/dataset",[12,14,16,11,13,15])

X = []
y = []

for landmarks, label in data:
    X.append(np.array(landmarks).flatten())  
    y.append(label)

X = np.array(X)
y = np.array(y)


# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# for k in range(1, 11):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=5)
#     print(f"k={k}, CV accuracy={scores.mean():.3f}")




# creation model KNN
knn = KNeighborsClassifier(n_neighbors=4)

# entrainer
knn.fit(X_train, y_train)

# evaluer
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(knn, "knn_pompe_model.pkl")