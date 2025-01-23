import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def train_model():
    data_path = "(ADD PATH)\data\machine_downtime.csv"
    data = pd.read_csv(data_path)

    data = data.dropna()
    data['Downtime'] = data['Downtime'].replace({'Machine_Failure': 1, 'No_Machine_Failure': 0})
    data = data.drop(['Date', 'Machine_ID', 'Assembly_Line_No'], axis=1)

    x = data.drop("Downtime", axis=1)
    y = data['Downtime']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, scaler, {"accuracy": accuracy, "f1_score": f1}
