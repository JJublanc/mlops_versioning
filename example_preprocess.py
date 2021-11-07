import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

target_col = 'target'

iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

features_cols = data.columns.drop(target_col)

data.to_csv('data/iris.csv', index=False)

def preprocess(data:pd.DataFrame,
               target_col:str,
               features_cols:list):

    X = data[features_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.4,
                                                        random_state = 42)

    X_test, X_evaluat, y_test, y_evaluate = train_test_split(X_test,
                                                             y_test,
                                                             test_size=0.4,
                                                             random_state=42)

    return X_train, X_test, X_evaluat, y_train, y_test, y_evaluate

if __name__ == "__main__":
    data = pd.read_csv("data/iris.csv")
    X_train, X_test, X_evaluat, y_train, y_test, y_evaluate = preprocess(data,
                                                                         target_col="target",
                                                                         features_cols=features_cols)