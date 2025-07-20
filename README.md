Este repositorio contiene el pipeline completo de análisis y modelado para **predecir la cancelación de reservas de hotel**.  

Incluye:

- **EDA detallado** (`notebooks/01_EDA.ipynb`): análisis univariante, bivariante y temporal.  
- **Preprocesado automático** (`src/preprocess.py`): imputación, encoding, escalado y SMOTE.  
- **Model Zoo** (`src/model_zoo/`):  
  - Regresión logística  
  - Árbol de decisión  
  - Random Forest  
  - Gradient Boosting: XGBoost, LightGBM y CatBoost  
  - MLP con Keras  
- **Entrenamiento y evaluación** (`src/train.py`, `src/evaluate.py`): selección de mejor modelo, métricas (F1, AUC-ROC, Precision/Recall), gráficas y modelo final serializado.

Este proyecto sigue buenas prácticas de ingeniería de datos y MLOps, y está diseñado para ser reproducible y modular.  
