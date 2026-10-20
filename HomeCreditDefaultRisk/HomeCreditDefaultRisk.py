import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import subprocess
import sys
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier

import os
from pathlib import Path

# Automatically get the current working directory
current_dir = Path(os.getcwd())

# Define file paths using pathlib
train_file = current_dir / "input" / "train.csv"
test_file = current_dir / "input" / "test.csv"
sample_sub_file = current_dir / "input" / "sample_submission.csv"

# Check if path exists
if train_file.exists() and test_file.exists() and sample_sub_file.exists():
    print("All files exist and path is correctly set.")
else:
    print("Some files are missing or path is not correctly set.")

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample_sub = pd.read_csv(sample_sub_file)

print(f"train shape: {train.shape}")
train.head(3)

print(f"test shape: {test.shape}")
test.head(3)

target = train["TARGET"].astype("int8").to_numpy()
test_ids = test["SK_ID_CURR"].copy()

X = train.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
X_test = test.drop(columns=["SK_ID_CURR"]).copy()

print(f"train shape: {train.shape}")
train.head(3)
print(f"test shape: {test.shape}")
test.head(3)

train.isnull().sum()
test.isnull().sum()
sns.countplot(data=train, x="TARGET")
plt.show()
sns.countplot(data=train, x="NAME_CONTRACT_TYPE")
plt.show()
plt.figure(figsize=(30, 10))
sns.countplot(data=train, x="ORGANIZATION_TYPE")
plt.tick_params(axis="x", rotation=90)
plt.show()

sns.displot(data=train, x="EXT_SOURCE_2")
plt.show()
sns.displot(data=train, x="AMT_INCOME_TOTAL")
plt.show()
sns.displot(data=train, x="AMT_INCOME_TOTAL", log_scale=10)
plt.show()
len(train["AMT_INCOME_TOTAL"].unique())
train["AMT_INCOME_TOTAL"].value_counts().head(10)

sns.displot(data=train, x="OWN_CAR_AGE")
plt.show()

def add_features(data):
    df = data.copy()

    df["EMPLOYMENT_365243_FLAG"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)
    df["CAR_AGE_OUTLIER_FLAG"] = (df["OWN_CAR_AGE"] >= 60).astype("int8")
    df["CAR_MISSING_FLAG"] = df["OWN_CAR_AGE"].isna().astype("int8")

    for col in ["CODE_GENDER", "ORGANIZATION_TYPE", "OCCUPATION_TYPE"]:
        if col in df.columns:
            df[col] = df[col].replace("XNA", np.nan)

    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = df["AMT_CREDIT"].replace(0, np.nan)
    annuity = df["AMT_ANNUITY"].replace(0, np.nan)
    goods = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    fam_members = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    days_birth = df["DAYS_BIRTH"].replace(0, np.nan)
    days_employed = df["DAYS_EMPLOYED"].replace(0, np.nan)


    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    external = df[ext_cols]
    df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCES_WEIGHTED"] = df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 1 + df["EXT_SOURCE_3"] * 3
    df["EXT_SOURCES_MIN"] = external.min(axis=1)
    df["EXT_SOURCES_MAX"] = external.max(axis=1)
    df["EXT_SOURCES_MEAN"] = external.mean(axis=1)
    df["EXT_SOURCES_NANMEDIAN"] = np.nanmedian(external.to_numpy(dtype="float64"), axis=1)
    df["EXT_SOURCES_VAR"] = external.var(axis=1)
    df["EXT_SOURCE_STD"] = external.std(axis=1)
    df["EXT_SOURCE_PRODUCT"] = external.prod(axis=1, min_count=1)
    df["EXT_SOURCE_COUNT"] = external.notna().sum(axis=1).astype("int8")
    df["EXT_SOURCE_1_x_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_SOURCE_2_x_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_1_x_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    w1 = df["EXT_SOURCE_1"].notna().astype(float) * 2.0
    w2 = df["EXT_SOURCE_2"].notna().astype(float) * 1.0
    w3 = df["EXT_SOURCE_3"].notna().astype(float) * 3.0
    df["EXT_SOURCES_WEIGHTED_NANSAFE"] = (df["EXT_SOURCE_1"].fillna(0) * 2.0 + df["EXT_SOURCE_2"].fillna(0) * 1.0 + df["EXT_SOURCE_3"].fillna(0) * 3.0) / (w1 + w2 + w3).replace(0, np.nan)

    df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / annuity
    df["ANNUITY_LENGTH_YEARS"] = df["CREDIT_TO_ANNUITY_RATIO"] / 12.0
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / credit
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / goods
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit
    df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["GOODS_FINANCED_EXTRA_RATIO"] = df["CREDIT_GOODS_DIFF"] / goods
    df["GOODS_TO_INCOME_RATIO"] = df["AMT_GOODS_PRICE"] / income

    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
    df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["DISPOSABLE_INCOME"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]
    df["DISPOSABLE_INCOME_PER_MEMBER"] = df["DISPOSABLE_INCOME"] / fam_members
    df["INCOME_TO_EMPLOYED_RATIO"] = df["AMT_INCOME_TOTAL"] / days_employed
    df["INCOME_TO_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / days_birth
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / fam_members
    df["CREDIT_PER_FAMILY_MEMBER"] = df["AMT_CREDIT"] / fam_members
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / fam_members
    df["ADULT_CNT"] = (df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]).clip(lower=1)
    df["INCOME_PER_ADULT"] = df["AMT_INCOME_TOTAL"] / df["ADULT_CNT"]

    df["EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / days_birth
    df["ID_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / days_birth
    df["REG_TO_BIRTH_RATIO"] = df["DAYS_REGISTRATION"] / days_birth
    df["PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / days_birth
    df["CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / days_birth
    df["CAR_TO_EMPLOYED_RATIO"] = df["OWN_CAR_AGE"] / days_employed
    df["PHONE_TO_EMPLOYED_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / days_employed

    df["EXT_MEAN_DIV_PAYMENT_RATE"] = df["EXT_SOURCES_MEAN"] / df["PAYMENT_RATE"].replace(0, np.nan)
    df["EXT_MEAN_x_AGE_DAYS"] = df["EXT_SOURCES_MEAN"] * (-df["DAYS_BIRTH"] / 365.25)
    df["EXT_MEAN_DIV_CREDIT_GOODS"] = df["EXT_SOURCES_MEAN"] / df["CREDIT_TO_GOODS_RATIO"].replace(0, np.nan)


    bureau_columns = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
    df["BUREAU_REQUESTS_TOTAL"] = df[bureau_columns].sum(axis=1, min_count=1)
    df["BUREAU_REQUESTS_MISSING"] = df[bureau_columns].isna().all(axis=1).astype("int8")
    recent_inq = df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0) + df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0)
    df["BUREAU_RECENT_SHARE"] = recent_inq / (df["BUREAU_REQUESTS_TOTAL"].fillna(0) + 1.0)
    df["CONTACT_FLAGS_SUM"] = (df["FLAG_MOBIL"] + df["FLAG_EMP_PHONE"] + df["FLAG_WORK_PHONE"]).astype("int8")
    df["REGION_POP_x_RATING"] = df["REGION_POPULATION_RELATIVE"] * df["REGION_RATING_CLIENT"]

    age_years = -df["DAYS_BIRTH"] / 365.0
    df["AGE_RANGE"] = pd.cut(age_years, bins=[0, 27, 40, 50, 65, 99], labels=["less_27", "27_40", "40_50", "50_65", "more_65"],).astype(object)

    df["CREDIT_TO_ANNUITY_GROUP"] = pd.cut(df["CREDIT_TO_ANNUITY_RATIO"], bins=[-np.inf, 0, 10, 15, 20, 25, 30, 50, np.inf], labels=["lower_0", "0_10", "10_15", "15_20", "20_25", "25_30", "30_50", "more_50"]).astype(object)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.copy()


def add_group_and_freq_features(train_df, test_df, y_train_arr):
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    cat_cols = [col for col in combined.columns
                if not pd.api.types.is_numeric_dtype(combined[col])]

    aguiar_groups = [(["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "AGE_RANGE"], "GROUP1"), (["CREDIT_TO_ANNUITY_GROUP", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"], "GROUP2"), (["ORGANIZATION_TYPE"], "GROUP3"),
        (["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"], "GROUP4")]
    aguiar_agg_cols = [("EXT_SOURCES_MEAN", "EXT_SOURCES"), ("AMT_INCOME_TOTAL", "INCOME"), ("CREDIT_TO_ANNUITY_RATIO", "CREDIT_TO_ANNUITY"), ("AMT_CREDIT", "CREDIT"), ("AMT_ANNUITY", "ANNUITY")]

    new_cols = {}
    for grp_keys, grp_prefix in aguiar_groups:
        temp_keys = [combined[k].astype(object).fillna("__MISSING__") for k in grp_keys]
        grp = combined.groupby(temp_keys, observed=True)
        for col_name, short_name in aguiar_agg_cols:
            med = grp[col_name].transform("median")
            std = grp[col_name].transform("std")
            new_cols[f"{grp_prefix}_{short_name}_MEDIAN"] = med.astype("float32")
            new_cols[f"{grp_prefix}_{short_name}_STD"] = std.astype("float32")
            new_cols[f"{grp_prefix}_{short_name}_DIFF"] = (combined[col_name] - med).astype("float32")

    for col in cat_cols:
        freq = combined[col].map(combined[col].value_counts(dropna=True))
        new_cols[f"{col}_COUNT_ENC"] = freq.astype("float32")

    combined = pd.concat([combined, pd.DataFrame(new_cols, index=combined.index)], axis=1)

    print("Imputing missing EXT_SOURCE_1/2/3 via Ridge Regression on demographic/financial features...")
    imp_predictors = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CREDIT_TO_ANNUITY_RATIO",
        "PAYMENT_RATE", "CREDIT_TO_GOODS_RATIO", "ANNUITY_TO_INCOME_RATIO", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "OWN_CAR_AGE", "GROUP1_EXT_SOURCES_MEDIAN", "GROUP2_EXT_SOURCES_MEDIAN", "GROUP3_EXT_SOURCES_MEDIAN"]
    X_imp_base = combined[imp_predictors].copy()
    for c in X_imp_base.columns:
        med_val = X_imp_base[c].median()
        X_imp_base[c] = X_imp_base[c].fillna(0.0 if np.isnan(med_val) else med_val)
    X_imp_mat = (X_imp_base - X_imp_base.mean()) / X_imp_base.std().replace(0, 1.0)
    X_imp_np = np.column_stack([np.ones(len(combined), dtype="float64"), X_imp_mat.to_numpy(dtype="float64")])

    imp_cols = {}
    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        obs_mask = combined[ext_col].notna().to_numpy()
        A_obs = X_imp_np[obs_mask]
        y_obs = combined.loc[obs_mask, ext_col].to_numpy(dtype="float64")
        reg_eye = 50.0 * np.eye(A_obs.shape[1], dtype="float64")
        reg_eye[0, 0] = 0.0
        weights = np.linalg.solve(A_obs.T @ A_obs + reg_eye, A_obs.T @ y_obs)
        pred_ext = np.clip(X_imp_np @ weights, 0.01, 0.99).astype("float32")
        imp_cols[f"{ext_col}_PRED"] = pred_ext
        imp_cols[f"{ext_col}_IMP"] = combined[ext_col].fillna(pd.Series(pred_ext)).astype("float32")

    imp_cols["IMPUTED_EXT_SOURCES_MEAN"] = (imp_cols["EXT_SOURCE_1_IMP"] + imp_cols["EXT_SOURCE_2_IMP"] + imp_cols["EXT_SOURCE_3_IMP"]) / 3.0
    imp_cols["IMPUTED_EXT_SOURCES_WEIGHTED"] = (imp_cols["EXT_SOURCE_1_IMP"] * 2.0 + imp_cols["EXT_SOURCE_2_IMP"] * 1.0 + imp_cols["EXT_SOURCE_3_IMP"] * 3.0)
    imp_cols["IMPUTED_EXT_SOURCES_PROD"] = (imp_cols["EXT_SOURCE_1_IMP"] * imp_cols["EXT_SOURCE_2_IMP"] * imp_cols["EXT_SOURCE_3_IMP"])
    combined = pd.concat([combined, pd.DataFrame(imp_cols, index=combined.index)], axis=1)

    print("Computing 1st-Place Solution 500-Nearest-Neighbors Target Mean (cKDTree)...")
    knn_cols = ["EXT_SOURCE_1_IMP", "EXT_SOURCE_2_IMP", "EXT_SOURCE_3_IMP", "CREDIT_TO_ANNUITY_RATIO"]
    knn_df = combined[knn_cols].copy()
    for c in knn_cols:
        knn_df[c] = knn_df[c].fillna(knn_df[c].median())
    knn_mat = ((knn_df - knn_df.mean()) / knn_df.std().replace(0, 1.0)).to_numpy(dtype="float32")

    X_knn_tr = knn_mat[:n_train]
    X_knn_te = knn_mat[n_train:]
    y_flt = y_train_arr.astype("float32")

    oof_knn_500 = np.zeros(n_train, dtype="float32")
    oof_knn_50 = np.zeros(n_train, dtype="float32")
    oof_knn_dist = np.zeros(n_train, dtype="float32")

    skf_knn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_knn.split(X_knn_tr, y_train_arr):
        tree = cKDTree(X_knn_tr[tr_idx])
        dists, nn_idx = tree.query(X_knn_tr[val_idx], k=500, eps=0.15, workers=-1)
        neighbor_targets = y_flt[tr_idx][nn_idx]
        oof_knn_500[val_idx] = neighbor_targets.mean(axis=1)
        oof_knn_50[val_idx] = neighbor_targets[:, :50].mean(axis=1)
        oof_knn_dist[val_idx] = dists.mean(axis=1)

    full_tree = cKDTree(X_knn_tr)
    dists_te, nn_te = full_tree.query(X_knn_te, k=500, eps=0.15, workers=-1)
    test_targets_nn = y_flt[nn_te]
    te_knn_500 = test_targets_nn.mean(axis=1)
    te_knn_50 = test_targets_nn[:, :50].mean(axis=1)
    te_knn_dist = dists_te.mean(axis=1)

    knn_out = pd.DataFrame({"KNN_500_TARGET_MEAN": np.concatenate([oof_knn_500, te_knn_500]).astype("float32"), "KNN_50_TARGET_MEAN": np.concatenate([oof_knn_50, te_knn_50]).astype("float32"), "KNN_500_DIST_MEAN":
         np.concatenate([oof_knn_dist, te_knn_dist]).astype("float32"),}, index=combined.index)
    combined = pd.concat([combined, knn_out], axis=1)


    for col in cat_cols:
        levels = pd.Index(combined[col].dropna().unique())
        combined[col] = pd.Categorical(combined[col], categories=levels)

    X_out = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test_out = combined.iloc[n_train:].copy().reset_index(drop=True)
    return X_out, X_test_out, cat_cols

X_model = add_features(X)
X_test_model = add_features(X_test)
X_model, X_test_model, categorical = add_group_and_freq_features(X_model, X_test_model, target)
print(f" Shapes X_model: {X_model.shape}, X_test_model: {X_test_model.shape}")


X_train, X_valid, y_train, y_valid = train_test_split(X_model, target, test_size=0.2, stratify=target, random_state=42)

def get_base_params():
  return{"objective": "binary", "metric": "auc", "boosting_type": "goss", "learning_rate": 0.01, "num_leaves": 54, "max_depth": 10, "subsample_for_bin": 240000, "reg_alpha": 0.436193, "reg_lambda": 0.479169,
        "colsample_bytree": 0.508716, "min_split_gain": 0.024766, "subsample": 1.0, "is_unbalance": False, "verbosity": -1, "force_col_wise": True, "num_threads": min(8, os.cpu_count() or 1), "seed": 737851,}
base_params = get_base_params()



def fit_candidate(params):
    valid_cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    training_data = lgb.Dataset(X_train, label=y_train, categorical_feature=valid_cat_cols, free_raw_data=False,)
    validation_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=valid_cat_cols, reference=training_data, free_raw_data=False,)

    model = lgb.train(params, training_data, num_boost_round=5000, valid_sets=[validation_data], callbacks=[lgb.early_stopping(150, verbose=False)],)
    predictions = model.predict(X_valid, num_iteration=model.best_iteration)
    score = roc_auc_score(y_valid, predictions)
    return model, score


candidates = []
for leaves in [31, 47, 54]:
    params = base_params.copy()
    params["num_leaves"] = leaves
    model, score = fit_candidate(params)
    candidates.append((score, params))
    print(f"{leaves} validation AUC = {score:.6f}")

best_score, best_params = max(candidates, key=lambda result: result[0])
print(f"Using {best_params['num_leaves']} leaves (Holdout AUC {best_score:.6f})")

def run_stratified_kfold(X_df, y_arr, X_test_df, params, output_dir, n_splits=5):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    valid_cat_cols = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    global_prior = float(np.mean(y_arr))
    smoothing = 30.0

    X_te_train = X_df.copy()
    X_te_test = X_test_df.copy()
    te_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in valid_cat_cols:
        X_te_train[f"{col}_OOF_TE"] = np.nan
    for tr_idx, val_idx in te_folds.split(X_df, y_arr):
        y_tr = y_arr[tr_idx]
        for col in valid_cat_cols:
            s_tr = X_df[col].iloc[tr_idx].astype(object).fillna("__MISSING__")
            s_va = X_df[col].iloc[val_idx].astype(object).fillna("__MISSING__")
            stats = pd.Series(y_tr).groupby(s_tr.to_numpy()).agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_prior) / (stats["count"] + smoothing)
            X_te_train.iloc[val_idx, X_te_train.columns.get_loc(f"{col}_OOF_TE")] = (s_va.map(smooth).fillna(global_prior).to_numpy(dtype="float32"))

    for col in valid_cat_cols:
        X_te_train[f"{col}_OOF_TE"] = X_te_train[f"{col}_OOF_TE"].astype("float32")
        s_all = X_df[col].astype(object).fillna("__MISSING__")
        s_te = X_test_df[col].astype(object).fillna("__MISSING__")
        stats_all = pd.Series(y_arr).groupby(s_all.to_numpy()).agg(["count", "mean"])
        smooth_all = (stats_all["count"] * stats_all["mean"] + smoothing * global_prior) / (stats_all["count"] + smoothing)
        X_te_test[f"{col}_OOF_TE"] = s_te.map(smooth_all).fillna(global_prior).astype("float32")

    X_num_train = X_te_train.copy()
    X_num_test = X_te_test.copy()
    for col in valid_cat_cols:
        X_num_train[col] = X_num_train[col].cat.codes.astype("int32")
        X_num_test[col] = X_num_test[col].cat.codes.astype("int32")

    goss_params = params.copy()
    goss_params["boosting_type"] = "goss"
    goss_params["learning_rate"] = 0.01

    gbdt_params = params.copy()
    gbdt_params["boosting_type"] = "gbdt"
    gbdt_params["learning_rate"] = 0.015
    gbdt_params["subsample"] = 0.85
    gbdt_params["subsample_freq"] = 1
    gbdt_params["num_leaves"] = 31

    oof_dict = {}
    test_dict = {}
    fold_scores = []
    best_iterations = []

    for name, cfg_params, seed in [("lgb_goss", goss_params, 737851), ("lgb_gbdt", gbdt_params, 42)]:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_p = np.zeros(len(X_te_train), dtype="float64")
        test_p = np.zeros(len(X_te_test), dtype="float64")

        for fold, (train_idx, valid_idx) in enumerate(folds.split(X_te_train, y_arr), start=1):
            dtrain = lgb.Dataset(X_te_train.iloc[train_idx], label=y_arr[train_idx], categorical_feature=valid_cat_cols)
            dvalid = lgb.Dataset(X_te_train.iloc[valid_idx], label=y_arr[valid_idx], categorical_feature=valid_cat_cols, reference=dtrain,)
            run_p = cfg_params.copy()
            run_p["seed"] = seed + fold

            model = lgb.train(run_p, dtrain, num_boost_round=6000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(150, verbose=False)])
            val_p = model.predict(X_te_train.iloc[valid_idx], num_iteration=model.best_iteration)
            oof_p[valid_idx] = val_p
            test_p += model.predict(X_te_test, num_iteration=model.best_iteration) / n_splits

            score = roc_auc_score(y_arr[valid_idx], val_p)
            fold_scores.append(score)
            best_iterations.append(model.best_iteration)
            model.save_model(str(model_dir / f"{name}_fold_{fold}.txt"))
            print(f"[{name.upper()} | Fold {fold}] AUC = {score:.6f} (best_iter = {model.best_iteration})")

        auc_val = roc_auc_score(y_arr, oof_p)
        print(f"--> {name.upper()} 5-Fold OOF AUC: {auc_val:.6f}")
        oof_dict[name] = oof_p
        test_dict[name] = test_p


    xgb_oof = np.zeros(len(X_num_train), dtype="float64")
    xgb_test = np.zeros(len(X_num_test), dtype="float64")
    xgb_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2024)
    xgb_params = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 60, "subsample": 0.8,
        "colsample_bytree": 0.55, "reg_alpha": 1.5, "reg_lambda": 10.0, "nthread": min(8, os.cpu_count() or 1)}

    dtest_xgb = xgb.DMatrix(X_num_test)
    for fold, (train_idx, valid_idx) in enumerate(xgb_folds.split(X_num_train, y_arr), start=1):
        dtr_xgb = xgb.DMatrix(X_num_train.iloc[train_idx], label=y_arr[train_idx])
        dva_xgb = xgb.DMatrix(X_num_train.iloc[valid_idx], label=y_arr[valid_idx])
        run_xp = xgb_params.copy()
        run_xp["seed"] = 2024 + fold

        bst = xgb.train(run_xp, dtr_xgb, num_boost_round=4000, evals=[(dva_xgb, "val")], early_stopping_rounds=150, verbose_eval=False,)
        best_ntree = getattr(bst, "best_iteration", bst.num_boosted_rounds() - 1)
        val_p = bst.predict(dva_xgb, iteration_range=(0, best_ntree + 1))
        xgb_oof[valid_idx] = val_p
        xgb_test += bst.predict(dtest_xgb, iteration_range=(0, best_ntree + 1)) / n_splits

        score = roc_auc_score(y_arr[valid_idx], val_p)
        fold_scores.append(score)
        best_iterations.append(best_ntree)
        print(f"[XGBOOST | Fold {fold}] AUC = {score:.6f} (best_iter = {best_ntree})")

    xgb_auc = roc_auc_score(y_arr, xgb_oof)
    print(f"--> XGBOOST 5-Fold OOF AUC: {xgb_auc:.6f}")
    oof_dict["xgb"] = xgb_oof
    test_dict["xgb"] = xgb_test

    cb_oof = np.zeros(len(X_num_train), dtype="float64")
    cb_test = np.zeros(len(X_num_test), dtype="float64")
    cb_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=999)

    for fold, (train_idx, valid_idx) in enumerate(cb_folds.split(X_num_train, y_arr), start=1):
        cb_model = CatBoostClassifier(iterations=4500, learning_rate=0.03, depth=6, l2_leaf_reg=10.0, eval_metric="AUC", random_seed=999 + fold, thread_count=min(8, os.cpu_count() or 1), verbose=False)
        cb_model.fit(X_num_train.iloc[train_idx], y_arr[train_idx], eval_set=(X_num_train.iloc[valid_idx], y_arr[valid_idx]), early_stopping_rounds=150, verbose=False,)
        val_p = cb_model.predict_proba(X_num_train.iloc[valid_idx])[:, 1]
        cb_oof[valid_idx] = val_p
        cb_test += cb_model.predict_proba(X_num_test)[:, 1] / n_splits

        score = roc_auc_score(y_arr[valid_idx], val_p)
        fold_scores.append(score)
        best_iterations.append(cb_model.get_best_iteration())
        print(f"[CATBOOST | Fold {fold}] AUC = {score:.6f} (best_iter = {cb_model.get_best_iteration()})")

    cb_auc = roc_auc_score(y_arr, cb_oof)
    print(f"--> CATBOOST 5-Fold OOF AUC: {cb_auc:.6f}")
    oof_dict["catboost"] = cb_oof
    test_dict["catboost"] = cb_test


    weights = {"lgb_goss": 0.35, "lgb_gbdt": 0.25, "xgb": 0.20, "catboost": 0.20}

    oof_prob_blend = sum(w * oof_dict[k] for k, w in weights.items())
    test_prob_blend = sum(w * test_dict[k] for k, w in weights.items())
    prob_auc = roc_auc_score(y_arr, oof_prob_blend)

    oof_rank_blend = sum(w * (rankdata(oof_dict[k]) / len(y_arr)) for k, w in weights.items())
    test_rank_blend = sum(w * (rankdata(test_dict[k]) / len(X_test_df)) for k, w in weights.items())
    rank_auc = roc_auc_score(y_arr, oof_rank_blend)

    print(f"4-Model Probability Blend OOF AUC: {prob_auc:.6f}")
    print(f"4-Model Percentile-Rank Blend OOF AUC: {rank_auc:.6f}")

    if rank_auc > prob_auc:
        final_oof = oof_rank_blend
        final_test = test_rank_blend
        final_oof_auc = rank_auc
        print(f"Selected Percentile-Rank Blend (OOF ROC-AUC = {final_oof_auc:.6f})")
    else:
        final_oof = oof_prob_blend
        final_test = test_prob_blend
        final_oof_auc = prob_auc
        print(f"Selected Probability Blend (OOF ROC-AUC = {final_oof_auc:.6f})")

    (output_dir / "validation_metrics.txt").write_text(f"Five-fold out-of-fold ROC AUC: {final_oof_auc:.6f}\n" + "Fold ROC AUCs: " + ", ".join(f"{s:.6f}" for s in fold_scores) + "\n"
                                                       + "Best iterations: " + ", ".join(map(str, best_iterations)) + "\n",encoding="utf-8")

    return final_oof, final_test, final_oof_auc


output_dir = current_dir / "output"
oof_predictions, test_predictions, oof_score = run_stratified_kfold(X_model, target, X_test_model, best_params, output_dir)

def build_submission(test_ids_series, raw_predictions, sample_sub_df):
    prediction_by_id = pd.Series(raw_predictions, index=test_ids_series.to_numpy())
    sample_ids = sample_sub_df["SK_ID_CURR"].to_numpy()
    if set(sample_ids) != set(test_ids_series.to_numpy()):
        raise ValueError("The sample submission IDs do not match test.csv")

    pred = sample_sub_df["SK_ID_CURR"].map(prediction_by_id).to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError("Some rows do not have a finite prediction")

    sub = sample_sub_df.copy()
    sub["TARGET"] = np.clip(pred, 0.0, 1.0)
    return sub


output_dir = current_dir / "output"
os.makedirs(output_dir, exist_ok=True)

# Specify the new output file path
output_file = output_dir / "submission.csv"

# Save the CSV file to the "output" directory
sample_sub.to_csv(output_file, index=False)
sample_sub = build_submission(test_ids, test_predictions, sample_sub)
sample_sub.head(10)