import pandas as pd
import matplotlib.pyplot as plt
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

#logistic regrestion , random forest, desion tree

def convert_to_binary_encoding(df):
    label_encoder = LabelEncoder()
    categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])



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

    rf_classifier = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
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

def Rf_gridsearch(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 15, None],
    }
    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")


if __name__ == '__main__':
    df = read_file("heart.csv")
    df = clean_data(df)
    convert_to_binary_encoding(df)

    X = df.drop('HeartDisease', axis=1) #x-axis represent the target
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    KNN(1,X_train, X_test, y_train, y_test)
    KNN(3,X_train, X_test, y_train, y_test)
    RF(X_train, X_test, y_train, y_test,50)
    RF(X_train, X_test, y_train, y_test,100)
    RF(X_train, X_test, y_train, y_test,200)
    RF(X_train, X_test, y_train, y_test,500)
    print("-------------------------------------------------------------------------")
    Rf_gridsearch(X_train, X_test, y_train, y_test)
=

