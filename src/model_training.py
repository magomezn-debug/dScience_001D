
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def obtener_modelos(random_state=42):

    modelos = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state
        ),

        'SVM': SVC(
            probability=True,
            random_state=random_state
        ),

        'KNN': KNeighborsClassifier()
    }

    return modelos


def crear_pipeline(preprocessor, modelo):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', modelo)
    ])

    return pipeline


def entrenar_modelo(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    return pipeline
