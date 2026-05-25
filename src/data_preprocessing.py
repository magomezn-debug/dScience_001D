
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def identificar_columnas(X):

    columnas_numericas = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    columnas_categoricas = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    return columnas_numericas, columnas_categoricas


def crear_preprocessor(columnas_numericas, columnas_categoricas):

    pipeline_numerico = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    pipeline_categorico = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', pipeline_numerico, columnas_numericas),
        ('cat', pipeline_categorico, columnas_categoricas)
    ])

    return preprocessor
