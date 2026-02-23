import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# (1) データ読み込み
# =========================================================
train = pd.read_csv('/input/playground-series-s6e2/train.csv')
test = pd.read_csv('/input/playground-series-s6e2/test.csv')

TARGET = 'Heart Disease'
ID_COL = 'id'
BASE = [c for c in train.columns if c not in [ID_COL, TARGET]]

train[TARGET] = train[TARGET].map({'Absence': 0, 'Presence': 1})

print(f"[1/6] Data loaded | Train: {train.shape} | Test: {test.shape}")

# =========================================================
# (2) 外部データなしで統計特徴量を追加（train だけで作成）
# =========================================================
# original を使わないため、train の統計量を使って
# mean/count 系の特徴量を train/test に付与する
STAT_FEATS = []
global_mean = train[TARGET].mean()

for col in BASE:
    mean_map = train.groupby(col)[TARGET].mean()
    count_map = train[col].value_counts().to_dict()

    train[f'train_mean_{col}'] = train[col].map(mean_map).fillna(global_mean)
    test[f'train_mean_{col}'] = test[col].map(mean_map).fillna(global_mean)
    train[f'train_count_{col}'] = train[col].map(count_map).fillna(0)
    test[f'train_count_{col}'] = test[col].map(count_map).fillna(0)

    STAT_FEATS.extend([f'train_mean_{col}', f'train_count_{col}'])

print(f"[2/6] Statistical features added (no external data) | {len(STAT_FEATS)} features")

# =========================================================
# (3) 手作り特徴量を追加
# =========================================================
for df in [train, test]:
    df['age_cholesterol'] = df['Age'] * df['Cholesterol']
    df['bp_age'] = df['BP'] * df['Age']
    df['vessels_thallium'] = df['Number of vessels fluro'] * df['Thallium']
    df['chest_pain_vessels'] = df['Chest pain type'] * df['Number of vessels fluro']
    df['hr_age_ratio'] = df['Max HR'] / (df['Age'] + 1)
    df['cholesterol_age_ratio'] = df['Cholesterol'] / (df['Age'] + 1)
    df['bp_ratio'] = df['BP'] / (df['Age'] + 1)
    df['heart_risk_score'] = (
        df['Thallium'] * 3
        + df['Chest pain type'] * 2
        + df['Number of vessels fluro'] * 2
    )
    df['thallium_sq'] = df['Thallium'] ** 2
    df['chest_pain_sq'] = df['Chest pain type'] ** 2

DERIVED = [
    'age_cholesterol',
    'bp_age',
    'vessels_thallium',
    'chest_pain_vessels',
    'hr_age_ratio',
    'cholesterol_age_ratio',
    'bp_ratio',
    'heart_risk_score',
    'thallium_sq',
    'chest_pain_sq',
]

print(f"[3/6] Derived features added | {len(DERIVED)} features")

# =========================================================
# (4) 学習用の X, y を作成
# =========================================================
FEATURES = BASE + STAT_FEATS + DERIVED

X = train[FEATURES].astype(np.float32)
y = train[TARGET]
X_test = test[FEATURES].astype(np.float32)

print(
    f"[4/6] X, y prepared | "
    f"{len(BASE)} base + {len(STAT_FEATS)} stats + {len(DERIVED)} derived = {len(FEATURES)} total"
)

# =========================================================
# (5) 相関を可視化
# =========================================================
corr_df = train[BASE + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(
    key=abs,
    ascending=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr_df.values]
ax.barh(corr_df.index, corr_df.values, color=colors, edgecolor='white', linewidth=0.7)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Correlation with Heart Disease', fontsize=11)
ax.set_title('Feature Correlation Analysis (Base Features)', fontsize=13, fontweight='bold')

for i, (val, name) in enumerate(zip(corr_df.values, corr_df.index)):
    ax.text(
        val + 0.01 if val >= 0 else val - 0.01,
        i,
        f'{val:.3f}',
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=9,
    )

plt.tight_layout()
plt.show()
print("[5/6] Correlation plot displayed")

# =========================================================
# (6) メモリ掃除
# =========================================================
gc.collect()
print("[6/6] Garbage collection completed")
