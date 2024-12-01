import pandas as pd
#from src.entity import DataTransformationConfig
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

def prepare():
    df = pd.read_csv('../data/raw/admission.csv')
    print(df.columns)
    assert (df.isnull().sum().sum() == 0)
    target = df['Chance of Admit ']
    feats = df.drop(['Serial No.', 'Chance of Admit '], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()    
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled).to_csv("../data/processed/X_train.csv", header=False, index=False)
    pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test.csv", header=False, index=False)
    pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", header=False, index=False)
    pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv", header=False, index=False)

if __name__ == '__main__':
    prepare()



