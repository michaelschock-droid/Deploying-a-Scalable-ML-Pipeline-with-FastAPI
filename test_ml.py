import pytest
# TODO: add necessary import
import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model
from ml.data import process_data, apply_label
import os


@pytest.fixture
def testing_data():
    """
    Create testing dataframe
    """
    data = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "workclass": ["Private", "Private", "Self-emp", "Federal-gov", "Private"],
        "fnlgt": [100000, 200000, 300000, 400000, 500000],
        "education": ["HS-grad", "Bachelors", "Masters", "Doctorate", "HS-grad"],
        "education-num": [9, 13, 14, 16, 9],
        "marital-status": ["Single", "Married", "Divorced", "Married", "Single"],
        "occupation": ["Craft-repair", "Prof-specialty", "Exec-managerial", "Prof-specialty", "Other-service"],
        "relationship": ["Own-child", "Husband", "Not-in-family", "Husband", "Unmarried"],
        "race": ["White", "White", "Black", "White", "Hispanic"],
        "sex": ["Male", "Male", "Female", "Male", "Female"],
        "capital-gain": [0, 10000, 0, 20000, 0],
        "capital-loss": [0, 0, 5000, 0, 0],
        "hours-per-week": [40, 45, 50, 60, 35],
        "native-country": ["United-States", "United-States", "Cuba", "United-States", "Mexico"],
        "salary": ["<=50K", ">50K", ">50K", ">50K", "<=50K"]
    })
    return data

@pytest.fixture
def testing_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

@pytest.fixture
def testing_split(testing_data,testing_features):
    X, y, _, _ = process_data(
        testing_data,
        categorical_features=testing_features,
        label="salary",
        training=True
    )
    return X, y, _



# TODO: implement the first test. Change the function name and input as needed
def test_train_model(testing_split):
    """
    Test the train_model function returns a valid model
    """
    X, y, _ = testing_split

    # Call the function
    model = train_model(X, y)

    # check the output model is valid
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')

# TODO: implement the second test. Change the function name and input as needed
def test_load_model(tmp_path):
    # Create a dummy model
    original_model = {"name": "test_model", "params": [1, 2, 3]}

    # Define a temporary path using pytest's tmp_path fixture
    model_path = tmp_path / "model.pkl"

    # Save then Load
    save_model(original_model, str(model_path))
    loaded_model = load_model(str(model_path))

    # Verify the dummy model survived the trip
    assert loaded_model == original_model
    assert os.path.exists(model_path)


# TODO: implement the third test. Change the function name and input as needed
def test_inference(testing_split,testing_data):
    """
    Test if the inference function returns predictions of the expected shape.
    """
    X, y, _ = testing_split
    # Train the model
    model = train_model(X, y)

    # Make predictions
    preds = inference(model, X)

    # Check if predictions have the expected shape and type
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(testing_data),)
