import pandas as pd
import matplotlib.pyplot as plt

# Example evaluation data for illustration
results = [
    {"Model": "Linear Regression", "RMSE": 145.6, "MAE": 98.3, "R2": 0.89},
    {"Model": "Lasso Regression", "RMSE": 138.4, "MAE": 94.2, "R2": 0.91},
    {"Model": "Random Forest", "RMSE": 78.6, "MAE": 65.3, "R2": 0.995},
    {"Model": "XGBoost", "RMSE": 81.2, "MAE": 66.9, "R2": 0.993}
]

df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(df)

df.plot.bar(x="Model", y=["RMSE", "MAE"], rot=45, figsize=(10, 6), title="Error Metrics by Model")
plt.ylabel("Error")
plt.grid(True)
plt.tight_layout()
plt.show()
