import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def EDA(df):
    # using statics to mean-median for numeric values / and mode with non-numeric values#
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])
    Mean = numeric_columns.mean()
    Median = numeric_columns.median()
    print("Mean values:")
    print(Mean)
    print("\nMedian values:")
    print(Median)

    non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64'])
    Mode = non_numeric_columns.mode().iloc[0]
    print("\nMode values:")
    print(Mode)
    print("dtypes",df.dtypes,"\n")

    features = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR",
                "ExerciseAngina", "Oldpeak", "ST_Slope"]
    target = "HeartDisease"

    # Split features into numerical and categorical
    numerical_features = df[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df[features].select_dtypes(include=['object']).columns

    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2,3, i)
        sns.histplot(data=df, x=feature, hue=target, multiple="stack", kde=True, palette="Set2", alpha=0.7)
        plt.title(f'Histogram for {feature}')
    plt.tight_layout()
    plt.show()

    # Plot bar plots for categorical features in subplots
    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(2, 3, i)
        sns.countplot(data=df, x=feature, hue=target, palette="Set2", alpha=0.7)
        plt.title(f'Bar Plot for {feature}')
    plt.tight_layout()
    plt.show()

    # here we should check the data first#
    print("befor handling the data ")
    print("isnull ",df.isnull().sum(),"\n")
    print("duplicated",df.duplicated().sum(),"\n")

    # here to check what is the best to fill the missing values -handle the data-#
    boxplot = df.boxplot(column=['Cholesterol'])
    boxplot.set_ylabel('Values')
    plt.show()

    # fill the missing values #
    df = df.fillna({"Sex": df['Sex'].mode()[0], "Cholesterol": df['Cholesterol'].median()})
    df = df.drop_duplicates()

    print("after handling the data ")
    print("isnull ",df.isnull().sum(),"\n")
    print("duplicated",df.duplicated().sum(),"\n")

    return df

def convert_to_binary_encoding(df):
    label_encoder = LabelEncoder()
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

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
    EDA(df)
    convert_to_binary_encoding(df)

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


