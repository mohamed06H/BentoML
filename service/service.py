"""This module defines a BentoML service that uses a Sklearn model to classify
Iris Species.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

BENTO_MODEL_TAG = "iris_clf:latest"


def get_model_by_tag(bento_model_tag):
    return bentoml.sklearn.get(bento_model_tag)


def get_model_in_prod(bento_model_name):
    list_models = bentoml.models.list(bento_model_name)
    for model in list_models:
        if model.info.labels.get('stage') == 'production':
            return model


# Get model from model store
model_to_run = get_model_in_prod("iris_clf")

print('-- model fetched from bentoml store : ', model_to_run.tag, model_to_run.info.labels, model_to_run.info.metadata)

classifier_runner = model_to_run.to_runner()

iris_service = bentoml.Service("iris_classifier", runners=[classifier_runner])

@iris_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return classifier_runner.predict.run(input_data)