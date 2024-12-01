import pandas as pd
from sklearn.linear_model import LinearRegression
import bentoml
from bentoml.io import NumpyNdarray

def train():

    X_train = pd.read_csv("../data/processed/X_train.csv").to_numpy()
    X_test = pd.read_csv("../data/processed/X_test.csv").to_numpy()
    y_train = pd.read_csv("../data/processed/y_train.csv").to_numpy()
    y_test = pd.read_csv("../data/processed/y_test.csv").to_numpy()
        
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    model_ref = bentoml.sklearn.save_model("universities_reg", reg)
    print(f"Modèle enregistré sous : {model_ref}")

if __name__ == '__main__':
    train()
    # bentoml models list
    # Tag                                Module           Size       Creation Time       
    # universities_reg:g433advp2kxlm6ip  bentoml.sklearn  1021.00 B  2024-12-01 10:51:27 