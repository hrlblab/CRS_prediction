import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


data_dir = "./results_auc"
label = "CRSwNP"
task_type = "binary"
task_name = f"{label}_{task_type}"


feature_sets = ["EO", "CY", "DM", "CM", "MH", "EO+CY", "EO+CY+DM", "EO+CY+DM+CM", "ALL"]
ref_feature = "EO+CY+DM+CM"


data_dict = {}
for feat in feature_sets:
    path = os.path.join(data_dir, f"{label}_{feat}_{task_type}.npy")
    if os.path.exists(path):
        auc = np.load(path)
        if isinstance(auc, np.ndarray) and len(auc) > 0:
            data_dict[feat] = auc


df = pd.DataFrame([(f, v) for f in data_dict for v in data_dict[f]], columns=["Feature", "AUC"])


ref_index = feature_sets.index(ref_feature)
p_results = []
for f in feature_sets:
    if f != ref_feature and f in data_dict:
        idx = feature_sets.index(f)
        _, p = ttest_rel(data_dict[f], data_dict[ref_feature])
        p_results.append((f, p, abs(idx - ref_index)))


all_entry = [r for r in p_results if r[0] == "ALL" and r[1] < 0.05]
rest = [r for r in p_results if r[0] != "ALL" and r[1] < 0.05]
rest_sorted = sorted(rest, key=lambda x: x[2])
p_results_sorted = all_entry + rest_sorted


plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")


ax = sns.boxplot(
    x="Feature", y="AUC", data=df, order=feature_sets,
    palette="Set2", showfliers=False, width=0.3
)


max_y = df["AUC"].max()
min_y = df["AUC"].min()
star_offset = 0.01
height_base = max_y + 0.01
line_height = 0.006
offset = 0.005
lines_used = 0

for (f, p, _) in p_results_sorted:
    i = feature_sets.index(f)
    x1, x2 = min(i, ref_index), max(i, ref_index)
    y = height_base + star_offset * lines_used
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1, c='k')
    ax.text((x1 + x2) / 2, y + offset, "*", ha='center', va='bottom', fontsize=10)
    lines_used += 1


y_min = min_y - 0.02
y_max = height_base + star_offset * (lines_used + 1.5)
plt.ylim(y_min, y_max)


plt.title(f"AUC Distribution: {task_name}", fontsize=13)
plt.ylabel("AUC", fontsize=11)
plt.xlabel("")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(f"{task_name}_auc.jpg", dpi=500)
# plt.show()
