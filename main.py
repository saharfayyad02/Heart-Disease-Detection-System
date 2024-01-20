import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def read_file(file):
    file_path = f"{file}"
    df = pd.read_csv(file_path)
    return df

def check_data(df):
    print("isnull ",df.isnull().sum(),"\n")
    print("dtypes",df.dtypes,"\n")
    print("duplicated",df.duplicated().sum(),"\n")
    print("describe",df.describe(),"\n")
    boxplot = df.boxplot(column=['Cholesterol'])
    boxplot.set_ylabel('Values')
    #plt.show()

def clean_data(df):
    data_frame = df.fillna({"Sex": df['Sex'].mode()[0], "Cholesterol": df['Cholesterol'].median()})
    data_frame = data_frame.drop_duplicates()
    return data_frame


def convert_to_binary_encoding(df):
    label_encoder = LabelEncoder()
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df

def KNN(k, X_train, X_test, y_train, y_test, metric='manhattan'):
    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Results for k={k} using {metric} distance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")

def RF(X_train, X_test, y_train, y_test,n_estimator):

    rf_classifier = RandomForestClassifier(n_estimators=n_estimator, random_state = 42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"RF for: {n_estimator}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")


def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning for Logistic Regression
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    logreg = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(logreg, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']

    # Use the best hyperparameter to train the model
    logreg = LogisticRegression(C=best_C, max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Results for Logistic Regression:")
    print(f"Best C: {best_C}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")


def SVM(X_train, X_test, y_train, y_test,c):
    clf = svm.SVC(kernel='linear', C=c)
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Results for SVM for {c}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")

if __name__ == '__main__':
    df = read_file("heart.csv")
    df = clean_data(df)
    df = convert_to_binary_encoding(df)
    df.to_csv('output.csv', index=False)

    X = df.drop('HeartDisease', axis=1) #x-axis represent the target
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

    print("--------------------------KNN-----------------------------------")
    #KNN(1,X_train, X_test, y_train, y_test)
    #KNN(3,X_train, X_test, y_train, y_test)
    # print("--------------------------RF------------------------------------")
    #RF(X_train, X_test, y_train, y_test,5)
    # RF(X_train, X_test, y_train, y_test,100)
    # RF(X_train, X_test, y_train, y_test,300)
    # RF(X_train, X_test, y_train, y_test,150)
    # print("--------------------------LR--------------------------------------")
    # LogisticRegressionModel(X_train, X_test, y_train, y_test)
    # print("-------------------------SVM-------------------------------------")
    # SVM(X_train, X_test, y_train, y_test,0.1)
    # SVM(X_train, X_test, y_train, y_test,1)
    # SVM(X_train, X_test, y_train, y_test,10)
    # SVM(X_train, X_test, y_train, y_test,100)

    #Rf_gridsearch(X_train, X_test, y_train, y_test)


