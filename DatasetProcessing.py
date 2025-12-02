import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 图 1：不同算法对比（含 MSE）
# -------------------------------
baseline_models = ['Bi-LANet', 'STS-PredNet', 'ConvLSTM', 'TFS-ACD (Ours)']
mape_baseline = [1.93, 2.23, 1.83, 1.04]
mae_baseline = [0.2064, 0.2912, 0.1952, 0.0754]
mse_baseline = [0.0797, 0.2220, 0.0645, 0.0082]
rmse_baseline = [0.2824, 0.4712, 0.2539, 0.0904]

x1 = np.arange(len(baseline_models))
width = 0.2

plt.figure(figsize=(10, 5))
plt.bar(x1 - 1.5*width, mape_baseline, width, label='MAPE (%)')
plt.bar(x1 - 0.5*width, mae_baseline, width, label='MAE')
plt.bar(x1 + 0.5*width, mse_baseline, width, label='MSE')
plt.bar(x1 + 1.5*width, rmse_baseline, width, label='RMSE')

plt.xlabel('Models')
plt.ylabel('Error')
plt.title('Performance Comparison with Baseline Models')
plt.xticks(x1, baseline_models, rotation=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("baseline_comparison_full.png", dpi=300)
plt.show()


# -------------------------------
# 图 2：消融实验对比（含 MSE）
# -------------------------------
ablation_models = ['w/o T', 'w/o F', 'w/o S', 'w/o Latent', 'w/o FFM', 'TFS-ACD']
mape_ablation = [1.61, 1.48, 1.37, 1.32, 1.19, 1.04]
mae_ablation = [0.1564, 0.1392, 0.1175, 0.1234, 0.1087, 0.0754]
mse_ablation = [0.0390, 0.0321, 0.0203, 0.0252, 0.0186, 0.0082]
rmse_ablation = [0.1975, 0.1791, 0.1426, 0.1589, 0.1365, 0.0904]

x2 = np.arange(len(ablation_models))

plt.figure(figsize=(11, 5))
plt.bar(x2 - 1.5*width, mape_ablation, width, label='MAPE (%)')
plt.bar(x2 - 0.5*width, mae_ablation, width, label='MAE')
plt.bar(x2 + 0.5*width, mse_ablation, width, label='MSE')
plt.bar(x2 + 1.5*width, rmse_ablation, width, label='RMSE')

plt.xlabel('Ablation Variants')
plt.ylabel('Error')
plt.title('Ablation Study on TFS-ACD Components')
plt.xticks(x2, ablation_models, rotation=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ablation_study_full.png", dpi=300)
plt.show()