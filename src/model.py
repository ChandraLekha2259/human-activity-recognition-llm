from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Train and return a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train.values.ravel()
    )

    return model