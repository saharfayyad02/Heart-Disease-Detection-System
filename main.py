import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,make_scorer,confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from itertools import product

def read_file(file):
    file_path = f"{file}"
    df = pd.read_csv(file_path)
    return df

def EDA(df):
    # using statics to mean-median for numeric values / and mode with non-numeric values#
    numerical_features = df.select_dtypes(include=['int64', 'float64'])
    Mean = numerical_features.mean()
    Median = numerical_features.median()
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

    means = df[numerical_features].mean()
    medians = df[numerical_features].median()
    modes = df[categorical_features].mode().iloc[0]
    color_palette = sns.color_palette("Set2", n_colors=len(numerical_features))

   # plot histogram for each numeric value -data visualization- #
    plt.figure(figsize=(10, 5))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=df[feature], kde=True, color=color_palette[i - 1], alpha=0.7)

        # Add mean, median, and mode lines
        plt.axvline(means[feature], color='darkred', linestyle='dashed', linewidth=2, label=f'Mean')
        plt.axvline(medians[feature], color='darkblue', linestyle='dashed', linewidth=2, label=f'Median')

        plt.title(f'Histogram for {feature}')
        plt.legend()

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 5))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(2, 3, i)

        # Check if the feature exists in the DataFrame
        if feature in df.columns:
            # Calculate the number of unique values for the current feature
            num_unique_values = df[feature].nunique()

            # Make sure the palette has enough colors for all unique values
            color_palette = sns.color_palette("Set2", num_unique_values)

            sns.countplot(data=df, x=feature, hue=feature, palette=color_palette, alpha=0.7, edgecolor='black', legend=False)
            # Add mode line
            plt.axvline(modes[feature], color='black', linestyle='dashed', linewidth=2, label=f'Mode')

            plt.title(f'Bar Plot for {feature}')
        else:
            print(f"Warning: {feature} is missing in the DataFrame.")

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

def convert_to_one_hot_encoding(df):
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    # Use get_dummies to perform one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns)
    df = df.astype(int)
    return df

def KNN(k, X_train, X_test, y_train, y_test, metric='manhattan'):
    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Results for k={k} using {metric} distance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")


def RF_testing(X_train, y_train):
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators' : [50,100,150,300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]  # Remove 'auto'
    }

    results = []

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    for max_features in param_grid['max_features']:
                        # Create a RandomForestClassifier with current hyperparameters
                        rf_classifier = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,random_state=42
                        )

                        # Train the model
                        rf_classifier.fit(X_train_new, y_train_new)

                        # Predictions on the test set
                        y_val_pred = rf_classifier.predict(X_val)

                        # Evaluate model performance on the test set
                        val_accuracy = accuracy_score(y_val, y_val_pred)
                        val_recall = recall_score(y_val, y_val_pred)

                        # Store the results
                        results.append({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_features': max_features,
                            'accuracy': val_accuracy,
                            'recall': val_recall
                        })
    best_accuracy = 0
    best_recall = 0
    best_hyperparameters = {}
    for result in results:
        accuracy = result['accuracy']
        recall = result['recall']

        # Check if the current combination has better accuracy or recall
        if accuracy > best_accuracy: #or (accuracy == best_accuracy and recall > best_recall):
            best_accuracy = accuracy
            best_recall = recall
            best_hyperparameters = {
                'n_estimators': result['n_estimators'],
                'max_depth': result['max_depth'],
                'min_samples_split': result['min_samples_split'],
                'min_samples_leaf': result['min_samples_leaf'],
                'max_features': result['max_features']
            }

    # Print the best hyperparameters and corresponding metrics
    print("Best Hyperparameters:")
    print(f"Best n_estimators: {best_hyperparameters['n_estimators']}")
    print(f"Best max_depth: {best_hyperparameters['max_depth']}")
    print(f"Best min_samples_split: {best_hyperparameters['min_samples_split']}")
    print(f"Best min_samples_leaf: {best_hyperparameters['min_samples_leaf']}")
    print(f"Best max_features: {best_hyperparameters['max_features']}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Recall: {best_recall:.4f}")

def RF(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(
        n_estimators=50, max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,random_state=42)

    rf_classifier.fit(X_train, y_train)
    y_test_pred = rf_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"Results of test data for RF :")
    print(f"Best Accuracy: {test_accuracy:.4f}")
    print(f"Best Recall: {test_recall:.4f}")
    print("\n****************************************************************\n")


def SVM_testing(X_train, y_train):

    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)

    # Define a parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1,1,10,100],
        'kernel': ['poly','linear','rbf','sigmoid'],
    }
    results = []
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
                    # Create an SVM classifier with current hyperparameters
                    svm_classifier = SVC(C=C, kernel=kernel, random_state=42)
                    # Train the model
                    svm_classifier.fit(X_train_new, y_train_new)
                    # Predictions on the test set
                    y_val_pred = svm_classifier.predict(X_val)
                    # Evaluate model performance on the test set
                    test_accuracy = accuracy_score(y_val, y_val_pred)
                    test_recall = recall_score(y_val, y_val_pred)

                    # Print results for the current hyperparameter combination
                    print(f"Hyperparameters: C={C}, kernel={kernel} for SVM ")
                    print(f"Accuracy: {test_accuracy:.4f}")
                    print(f"Recall: {test_recall:.4f}")

                    # Store the results
                    results.append({
                        'hyperparameters': {'C': C, 'kernel': kernel},
                        'accuracy': test_accuracy,
                        'recall': test_recall
                    })

    # Find the best hyperparameters based on accuracy and recall
    best_result = max(results, key=lambda x: (x['accuracy'], x['recall']))

    # Print the best hyperparameters and corresponding metrics
    print("\nBest Hyperparameters for SVM:")
    print(best_result['hyperparameters'])
    print(f"Best Accuracy: {best_result['accuracy']:.4f}")
    print(f"Best Recall: {best_result['recall']:.4f}")

    return best_result['hyperparameters']

def SVM(X_train, X_test, y_train, y_test):
    # Create an SVM classifier with current hyperparameters
    svm_classifier = SVC(C=0.1, kernel='linear', random_state=42)
    # Train the model
    svm_classifier.fit(X_train, y_train)
    # Predictions on the test set
    y_test_pred = svm_classifier.predict(X_test)
    # Evaluate model performance on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # Print results for the current hyperparameter combination
    print(f"Results of test data for SVM")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Recall: {test_recall:.4f}")


if __name__ == '__main__':
    df = read_file("heart.csv")
    df = EDA(df)
    df = convert_to_one_hot_encoding(df)
    df.to_csv('output.csv',index=False)

    X = df.drop('HeartDisease', axis=1) #x-axis represent the target
    y = df['HeartDisease']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

    print("--------------------------KNN-----------------------------------")
    KNN(1,X_train, X_test, y_train, y_test)
    KNN(3,X_train, X_test, y_train, y_test)
    print("--------------------------RF------------------------------------")
    #RF_testing(X_train, y_train)
    RF(X_train, X_test, y_train, y_test)
    print("-------------------------SVM-------------------------------------")
    #SVM_testing(X_train, y_train)
    SVM(X_train, X_test, y_train, y_test)

