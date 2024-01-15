import pandas as pd
import matplotlib.pyplot as plt
def read_file(file):
    file_path = f"{file}"
    df = pd.read_csv(file_path)

    return df

def check_data(df):
    print("isnull ",df.isnull().sum(),"\n")
    print("dtypes",df.dtypes,"\n")
    print("duplicated",df.duplicated().sum(),"\n")
    print("describe",df.describe(),"\n")
    boxplot = df.boxplot(column=['Age'])
    boxplot.set_xlabel('Age')
    boxplot.set_ylabel('Values')
    #plt.show()

def clean_data(df):
    data_frame = df.fillna({"Estrogen Status": df['Estrogen Status'].mode()[0], "Age": df['Age'].mean()})
    data_frame = data_frame.drop_duplicates()
    return data_frame


if __name__ == '__main__':
    df = read_file("C:\\Users\\user\\Desktop\\Breast_Cancer.csv")
    df = clean_data(df)
    check_data(df)

    Age = df['Age']
    Race = df['Race']
    Marital_Status = df['Marital Status']
    T_Stage = df['T Stage ']
    N_Stage = df['N Stage']
    six_th_Stage = df['6th Stage']
    differentiate = df['differentiate']
    Grade = df['Grade']
    A_Stage = df['A Stage']
    Tumor_Size = df['Tumor Size']
    Estrogen_Status = df['Estrogen Status']
    Progesterone_Status = df['Progesterone Status']
    Regional_Node_Examined = df['Regional Node Examined']
    Reginol_Node_Positive = df['Reginol Node Positive']
    Survival_Months = df['Survival Months']
    Status = df['Status']


