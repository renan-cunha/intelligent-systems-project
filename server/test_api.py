import cloudpickle as cp
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
import os

def load_model():
    with open(os.getenv("MODEL_PATH"), "rb") as file:
        return cp.load(file)

def load_data():
    data = pd.read_csv(os.getenv("TEST_PRODUCTS_PATH"))
    string_columns = data.select_dtypes("object").columns.tolist()
    data.loc[:, string_columns] = data.loc[:, string_columns].fillna("")
    return data


def test_check_columns():   
    data = load_data()
    expected = ['title', 'query', 'concatenated_tags']

    assert np.all(pd.Series(expected).isin(data.columns))

def test_load_pipeline_model():
    model = load_model()
    expected = Pipeline
    assert expected == model.__class__

def test_column_concatenation():
    data = load_data()
    model = load_model()

    expected = data["title"] + " " + data["concatenated_tags"]
    assert expected.equals(model["preprocessor"]["text_column_concatenation"].transform(data))

def test_preprocessor_pipeline_output_class():
    data = load_data()
    model = load_model()

    expected = csr_matrix
    assert expected == model["preprocessor"].transform(data).__class__

def test_pipeline_predict():
    data = load_data()
    model = load_model()
    labels = model.classes_

    y_true = data["category"]
    y_proba = model.predict_proba(data)

    assert roc_auc_score(y_true, y_proba, multi_class="ovr") > 0.97
