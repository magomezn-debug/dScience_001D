
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)


def optimizar_logistic(pipeline, X_train, y_train):

    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2']
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


def optimizar_decision_tree(pipeline, X_train, y_train):

    param_grid = {
        'model__max_depth': [3, 5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


def optimizar_svm(pipeline, X_train, y_train):

    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


def optimizar_knn(pipeline, X_train, y_train):

    param_dist = {
        'model__n_neighbors': [3, 5, 7, 9, 11],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return random_search
