from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from program.cross_validation import kFold_cross_validation_knn
from program.hyperparameter_optimization import hyper_opt_knn


def knn_classifier(df, col_list, hyp_opt: bool = False):
    X_food = df[col_list]
    y_food = df['nutriscore_grade']
    # hyperparameter optimization
    if hyp_opt:
        print("> Hyperparameter optimization started...")
        hypers = hyper_opt_knn(X_food, y_food)
    # cross-validation
    kFold_cross_validation_knn(X_food, y_food, hypers, 'classifier', 26)
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, test_size=0.2, random_state=200)
    # normalizing
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)  # features normalization (test set)
    knn = KNeighborsClassifier(n_neighbors=15)  # creo un knnClassifier
    knn.fit(X_train_scaled, y_train)  # fit del knn
    y_pred = knn.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Checking performance on the training set
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train_scaled, y_train)))
    # Checking performance on the test set
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test)))

def knn_regressor(df, col_list, hyp_opt: bool = False):
    X_food = df[col_list]
    y_food = df['nutriscore_grade']
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 7, 'weights': 'distance'}
    # hyperparameter optimization
    if hyp_opt:
        print("> Hyperparameter optimization started...")
        hypers = hyper_opt_knn(X_food, y_food)
    # cross-validation
    kFold_cross_validation_knn(X_food, y_food, hypers, 'regressor', 26)
