import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


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

if __name__ == '__main__':
    df = read_file("heart.csv")
    df = clean_data(df)
    convert_to_binary_encoding(df)

    X = df.drop('HeartDisease', axis=1) #x-axis represent the target
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    KNN(1,X_train, X_test, y_train, y_test)
    KNN(3,X_train, X_test, y_train, y_test)


    # Age = df['Age']
    # Sex = df['Sex']
    # ChestPainType = df['ChestPainType']
    # RestingBP = df['RestingBP']
    # Cholesterol = df['Cholesterol']
    # FastingBS = df['FastingBS']
    # RestingECG = df['RestingECG']
    # MaxHR = df['MaxHR']
    # ExerciseAngina = df['ExerciseAngina']
    # Oldpeak = df['Oldpeak']
    # ST_Slope = df['ST_Slope']
    # HeartDisease = df['HeartDisease']

