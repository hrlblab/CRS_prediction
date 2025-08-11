import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. data preprocess
# ----------------------------
file_path = 'C:/hrlblab/CRSproject/yifei_naweed_CRS-project-main/Sep2024respositorydeID2.xlsx'
data = pd.read_excel(file_path, sheet_name='out-s1')


data_clean = data.dropna(subset=['prior_surgery'])


X_task1 = data_clean.drop(columns=['phenotype', 'prior_surgery', 'number_prior_surgeries'])
y_task1 = data_clean['prior_surgery']


X_task1 = X_task1.apply(lambda col: col.fillna(col.mean()), axis=0)


scaler = StandardScaler()
X_task1 = pd.DataFrame(scaler.fit_transform(X_task1), columns=X_task1.columns)

# ----------------------------
# 2. Leave-One-Out
# ----------------------------
loo = LeaveOneOut()
auc_macro_scores = []
# auc_weighted_scores = []

for seed in range(100):
    y_true_all = []
    y_prob_all = []
    print(seed)

    for train_index, test_index in loo.split(X_task1):
        X_train, X_test = X_task1.iloc[train_index], X_task1.iloc[test_index]
        y_train, y_test = y_task1.iloc[train_index], y_task1.iloc[test_index]


        forest_clf = RandomForestClassifier(random_state=seed)
        forest_clf.fit(X_train, y_train)


        y_prob = forest_clf.predict_proba(X_test)[0]
        y_true_all.append(y_test.values[0])
        y_prob_all.append(y_prob)



    y_true_all = np.array(y_true_all)
    y_true_all_binarized = np.vstack((1 - y_true_all, y_true_all)).T
    y_prob_all = np.array(y_prob_all)
    auc_macro = roc_auc_score(y_true_all_binarized, y_prob_all, average='macro', multi_class='ovr')
    # auc_weighted = roc_auc_score(y_true_all_binarized, y_prob_all, average='weighted', multi_class='ovr')
    auc_macro_scores.append(auc_macro)
    # auc_weighted_scores.append(auc_weighted)


mean_auc_macro = np.mean(auc_macro_scores)
# mean_auc_weighted = np.mean(auc_weighted_scores)
print(f"Mean AUC (macro) over 100 runs: {mean_auc_macro:.4f}")
# print(f"Mean AUC (weighted) over 100 runs: {mean_auc_weighted:.4f}")


from scipy import stats


auc_macro_scores = np.array(auc_macro_scores)
mean_auc = np.mean(auc_macro_scores)
sem_auc = stats.sem(auc_macro_scores)
ci_low, ci_high = stats.t.interval(0.95, len(auc_macro_scores)-1, loc=mean_auc, scale=sem_auc)

print(f"\n--- AUC Macro Summary ---")
print(f"Mean AUC: {mean_auc:.4f}")
print(f"95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")

plt.hist(auc_macro_scores, bins=20, color='skyblue', edgecolor='black')
plt.axvline(mean_auc, color='red', linestyle='--', label=f'Mean AUC = {mean_auc:.4f}')
plt.axvline(ci_low, color='green', linestyle='--', label='95% CI Lower')
plt.axvline(ci_high, color='green', linestyle='--', label='95% CI Upper')
plt.title("AUC Macro Distribution (Original Model)")
plt.xlabel("AUC Macro Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

