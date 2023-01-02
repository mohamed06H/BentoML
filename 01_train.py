import warnings

import bentoml


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_model():
    """Trains a model.

    Args:

    Returns:
        The trained model.
    """
    model = KNeighborsClassifier()

    # Load in the data
    X, y = fetch_openml("iris", version=1, as_frame=True, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score


def main():
    """Trains a model for classifying Iris dataset."""

    # suppress future warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    model , score = train_model()

    # save model to bentoml /> bentoml models list
    bento_model = bentoml.sklearn.save_model("iris_clf",
                                             model,
                                             labels={
                                                 "owner": "mohamed",
                                                 "stage": "production"
                                             },
                                             metadata={
                                                 "accuracy": score
                                             }
                                    )
    print('bento_model saved \n', bento_model.tag, '\n', bento_model.info.metadata)


if __name__ == "__main__":
    main()
