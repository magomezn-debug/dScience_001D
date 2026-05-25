
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluar_modelo(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps['model'], 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan

    resultados = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }

    return resultados


def obtener_matriz_confusion(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)

    return confusion_matrix(y_test, y_pred)


def generar_classification_report(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)

    return classification_report(
        y_test,
        y_pred,
        zero_division=0
    )


def comparar_modelos(resultados):

    df = pd.DataFrame(resultados)

    return df.sort_values(
        by='f1_score',
        ascending=False
    )
