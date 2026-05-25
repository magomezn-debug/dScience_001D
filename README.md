# Proyecto de Modelado - Predicción de Abandono de Clientes

## Descripción del proyecto

Este proyecto corresponde a la Evaluación 2 de Programación para Ciencia de Datos.

El objetivo principal es desarrollar modelos de Machine Learning capaces de predecir el abandono de clientes utilizando información demográfica, económica y de comportamiento.

## Estructura sugerida del proyecto

```text
proyecto_modelado/
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
│
├── data/
│   └── dataset_clientes.csv
│
├── models/
│   └── trained_models/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── README.md
└── requirements.txt
```

## Modelos utilizados

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- KNN

## Técnicas utilizadas

- Pipelines
- ColumnTransformer
- StandardScaler
- One Hot Encoding
- Cross Validation
- GridSearchCV
- RandomizedSearchCV

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Orden recomendado de ejecución

1. `01_exploratory_analysis.ipynb`
2. `02_supervised_modeling.ipynb`
3. `03_model_evaluation.ipynb`
4. `04_hyperparameter_optimization.ipynb`
5. `05_final_analysis.ipynb`

## Objetivo de negocio

Detectar clientes con riesgo de abandono para apoyar decisiones preventivas, reducir pérdidas y mejorar la retención de clientes.
