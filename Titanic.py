# import numpy as np
import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifact


# Assign values to columns and get dummies
def imputation(df, col_name):
    col = df[col_name].unique()
    i = 0
    coldict = {}
    for c in col:
        coldict[c] = i
        i = i + 1
    df.replace({col_name: coldict}, inplace=True)
    df = pd.get_dummies(df, columns=[col_name])
    return df


# def plot_analysis():
#     titanic_attr = titanic_df.iloc[:, 0:10]
#     sns.pairplot(titanic_attr, diag_kind='kde')

def split_data():
    test_size = 0.20
    train, test = train_test_split(titanic_df, test_size=test_size, random_state=1)
    log_param("Test size", test_size)

    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]
    X_test = test.drop("PassengerId", axis=1).copy()
    y_test = test["Survived"]
    X_train.shape, y_train.shape, X_test.shape
    log_param("Train shape", X_train.shape)
    return X_train, y_train, X_test, y_test


# Train with Gaussian Naive Bayes
def gaussian_nb(X_train, y_train, X_test, y_test):
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)

    y_pred = gaussian_nb.predict(X_test)

    gnb_score = gaussian_nb.score(X_train, y_train)
    log_metric("GaussianNB Accuracy", gnb_score)
    # mlflow.log_metric("Precision for this run", precision_score(y_test, y_pred, average="binary"))
    # mlflow.log_metric("Recall for this run", recall_score(y_test, y_pred, average="binary"))
    # mlflow.log_metric("f1 for this run", f1_score(y_test, y_pred, average="binary"))

    mlflow.sklearn.log_model(gaussian_nb, "Gaussian Naive Bayes")


# Train with Logistic Regression
def logistic_reg(X_train, y_train, X_test, y_test):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    y_pred = logistic_reg.predict(X_test)

    lr_score = logistic_reg.score(X_train, y_train)
    log_metric("Logistic Regression Accuracy", lr_score)
    mlflow.log_metric("Logistic Regression Precision", precision_score(y_test, y_pred, average="binary"))
    mlflow.log_metric("Logistic Regression Recall", recall_score(y_test, y_pred, average="binary"))
    mlflow.log_metric("Logistic Regression f1 score", f1_score(y_test, y_pred, average="binary"))

    mlflow.sklearn.log_model(logistic_reg, "Logistic Regression")


# Train with KNeighbours Classifier
def knn(X_train, y_train, X_test, y_test, neigb):
    knn = KNeighborsClassifier(n_neighbors=neigb)
    log_param("n_neighbors", neigb)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    knn_score = knn.score(X_train, y_train)
    log_metric("KNN Accuracy", knn_score)
    # mlflow.log_metric("Precision for this run", precision_score(y_test, y_pred, average="binary"))
    # mlflow.log_metric("Recall for this run", recall_score(y_test, y_pred, average="binary"))
    # mlflow.log_metric("f1 for this run", f1_score(y_test, y_pred, average="binary"))

    mlflow.sklearn.log_model(knn, "KNeighbours Classifier")


# Train with Random Forest
def random_forest(X_train, y_train, X_test, y_test, n_est, max_dep, crit="gini"):
    random_forest = RandomForestClassifier(n_estimators=n_est, criterion=crit, max_depth=max_dep)
    log_param("n_estimators", n_est)
    log_param("criterion", crit)
    log_param("max_depth", max_dep)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)

    rf_score = random_forest.score(X_train, y_train)
    log_metric("Random Forest Accuracy", rf_score)
    log_metric("Random Forest Precision", precision_score(y_test, y_pred, average="binary"))
    log_metric("Random Forest Recall", recall_score(y_test, y_pred, average="binary"))
    log_metric("Random Forest f1 score", f1_score(y_test, y_pred, average="binary"))

    mlflow.sklearn.log_model(random_forest, "Random Forest Classifier")


if __name__ == '__main__':
    print('Starting the experiments')

    ##mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name='Titanic')
    titanic_df = pd.read_csv('Titanic+Data+Set.csv')
    log_artifact("Titanic+Data+Set.csv")

    # Check for null vals and drop unnecessary cols
    # titanic_df.isnull().sum()
    titanic_df = titanic_df.drop("Cabin", axis=1)
    titanic_df = titanic_df.drop("Name", axis=1)
    titanic_df = titanic_df.drop("Ticket", axis=1)

    # Fill null vals
    # titanic_df['Embarked'].describe()
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
    # titanic_df.isnull().sum()
    # titanic_df['Age'].describe()
    titanic_df['Age'] = titanic_df['Age'].fillna(28)
    # titanic_df.isnull().sum()

    # Create family feature from Parch & SibSp
    titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"]
    titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
    titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0
    # drop Parch & SibSp
    titanic_df = titanic_df.drop(['SibSp', 'Parch'], axis=1)

    # Separate columns
    titanic_df = imputation(titanic_df, 'Sex')
    titanic_df = imputation(titanic_df, 'Embarked')
    titanic_df = pd.get_dummies(titanic_df, columns=['Pclass', 'Family'])
    # titanic_df.dtypes

    # Plot features and analyze
    # plot_analysis()

    X_train, y_train, X_test, y_test = split_data()
    gaussian_nb(X_train, y_train, X_test, y_test)
    logistic_reg(X_train, y_train, X_test, y_test)
    # knn(X_train, y_train, X_test, y_test, 3)
    # knn(X_train, y_train, X_test, y_test, 6)
    knn(X_train, y_train, X_test, y_test, 10)
    # random_forest(X_train, y_train, X_test, y_test, 100, 8, "gini")
    random_forest(X_train, y_train, X_test, y_test, 120, 10, "entropy")
    # random_forest(X_train, y_train, X_test, y_test, 100, 4, "gini")
