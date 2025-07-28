import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_DIR, MODEL_NAMES
from src.config import NUM_FEATURES, CAT_FEATURES

# Carga el modelo Random Forest
model = joblib.load(MODEL_DIR / "random_forest.joblib")

# Carga el preprocesador
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

# Obtener nombres de features después del preprocesado
num_features = NUM_FEATURES
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = cat_encoder.get_feature_names_out(CAT_FEATURES)

# Unir nombres finales
final_feature_names = np.concatenate([num_features, cat_feature_names])

# Obtener importancias del modelo
importances = model.feature_importances_
importances_df = pd.DataFrame({
    "feature": final_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(20)

# Graficar
plt.figure(figsize=(10, 6))
plt.barh(importances_df["feature"], importances_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()

# Guardar figura
plt.savefig("reports/figures/feature_importance_rf.png")
plt.show()
