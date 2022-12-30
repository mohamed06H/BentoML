import warnings
from typing import Tuple
import json

import numpy as np
import requests
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SERVICE_URL = "http://localhost:3000/classify"


def sample_random_iris_data_point() -> Tuple[np.ndarray, np.ndarray]:

    # Load in the data
    X, y = fetch_openml("iris", version=1, as_frame=True, return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    random_index = np.random.randint(0, len(X_test))

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    random_test_tuple = X_test[random_index]
    random_test_tuple = np.expand_dims(random_test_tuple, 0)

    return random_test_tuple, y_test[random_index]


def request_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():

    # suppress future warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    input_data, expected_output = sample_random_iris_data_point()
    prediction = request_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()
