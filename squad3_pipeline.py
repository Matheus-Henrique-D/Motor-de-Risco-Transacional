"""
================================================================================
SQUAD 3 — Motor de Risco Transacional Multi-Tabelas
Home Credit Default Risk | Pipeline Sênior Completo
================================================================================
Etapas:
  1. Carregamento e diagnóstico dos dados
  2. Limpeza e tratamento (anomalias, nulos estruturais, encoding)
  3. Feature Engineering com janelas temporais (3, 6, 12 meses)
  4. Join seguro (sem data leakage)
  5. Modelagem LightGBM com StratifiedKFold
  6. Análise financeira: Profit Curve e threshold ótimo de negócio
  7. Exportação de gráficos e submission
================================================================================
"""

import os, warnings, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

# ── Caminhos ──────────────────────────────────────────────────────────────────
DATA_DIR   = r"c:\Users\joker\OneDrive\Área de Trabalho\Atividade final Julio\home-credit-default-risk"
OUTPUT_DIR = r"c:\Users\joker\OneDrive\Área de Trabalho\Atividade final Julio\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Premissas financeiras de negócio ─────────────────────────────────────────
TAXA_RECUPERACAO  = 0.20   # % do crédito recuperado em caso de inadimplência
PRAZO_MEDIO_MESES = 24     # proxy de prazo para calcular receita de FP
RANDOM_STATE      = 42
N_SPLITS          = 5

DIVIDER = "=" * 72

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def mem_usage(df):
    """Retorna uso de memória formatado."""
    mem = df.memory_usage(deep=True).sum() / 1024**2
    return f"{mem:.1f} MB"

def reduce_mem(df):
    """Reduz tipos de dados para economizar RAM."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    return df

def null_report(df, name):
    """Imprime relatório rápido de nulos."""
    total_null = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    pct = total_null / total_cells * 100
    print(f"  {name:30s} | shape={str(df.shape):18s} | nulls={total_null:,} ({pct:.1f}%) | mem={mem_usage(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════
print(DIVIDER)
print("SQUAD 3 | Motor de Risco Transacional | Home Credit Default Risk")
print(DIVIDER)
print("\n[BLOCO 1] Carregando dados...\n")

app_train = pd.read_csv(os.path.join(DATA_DIR, "application_train.csv"))
app_test  = pd.read_csv(os.path.join(DATA_DIR, "application_test.csv"))
bureau    = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"))
bb        = pd.read_csv(os.path.join(DATA_DIR, "bureau_balance.csv"))
prev      = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"))
pos       = pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"))
inst      = pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"))
cc        = pd.read_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"))

print(f"  {'Tabela':30s} | {'Shape':18s} | {'Nulos (%)':15s} | Memória")
print(f"  {'-'*70}")
null_report(app_train, "application_train")
null_report(app_test,  "application_test")
null_report(bureau,    "bureau")
null_report(bb,        "bureau_balance")
null_report(prev,      "previous_application")
null_report(pos,       "POS_CASH_balance")
null_report(inst,      "installments_payments")
null_report(cc,        "credit_card_balance")

print(f"\n  Desbalanceamento TARGET:")
vc = app_train["TARGET"].value_counts()
print(f"    Classe 0 (Adimplente) : {vc[0]:>7,} ({vc[0]/len(app_train)*100:.1f}%)")
print(f"    Classe 1 (Inadimplente): {vc[1]:>7,} ({vc[1]/len(app_train)*100:.1f}%)")
print(f"    Razão de desbalanceamento: 1:{vc[0]/vc[1]:.0f}")

# Reduz memória
for df in [app_train, app_test, bureau, bb, prev, pos, inst, cc]:
    reduce_mem(df)
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — LIMPEZA E TRATAMENTO DA TABELA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 2] Limpeza e tratamento da tabela principal...\n")

def limpar_application(df):
    df = df.copy()

    # ── Anomalias documentadas ───────────────────────────────────────────────
    # DAYS_EMPLOYED = 365243 é código para "não empregado / aposentado"
    df["FLAG_DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # ── Conversão temporal (dias → anos) ─────────────────────────────────────
    df["AGE_YEARS"]          = (df["DAYS_BIRTH"]        / -365.25).astype(np.float32)
    df["YEARS_EMPLOYED"]     = (df["DAYS_EMPLOYED"]     / -365.25).astype(np.float32)
    df["YEARS_REGISTRATION"] = (df["DAYS_REGISTRATION"] / -365.25).astype(np.float32)
    df["YEARS_ID_PUBLISH"]   = (df["DAYS_ID_PUBLISH"]   / -365.25).astype(np.float32)

    # Removemos as originais em dias (evitar redundância e multicolinearidade)
    df.drop(columns=["DAYS_BIRTH", "DAYS_REGISTRATION",
                      "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"],
            errors="ignore", inplace=True)

    # ── Ratios financeiros (features de alto valor preditivo) ─────────────────
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1e-5)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    df["CREDIT_GOODS_RATIO"]   = df["AMT_CREDIT"]  / (df["AMT_GOODS_PRICE"]  + 1e-5)
    df["INCOME_PER_MEMBER"]    = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1e-5)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
    df["LOAN_VALUE_GAP"]       = df["AMT_CREDIT"]  - df["AMT_GOODS_PRICE"]

    # ── Flags de nulo (nulo é informação!) ────────────────────────────────────
    colunas_criticas_null = [
        "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL",
        "YEARS_EMPLOYED", "CNT_FAM_MEMBERS", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    for col in colunas_criticas_null:
        if col in df.columns:
            df[f"FLAG_{col}_NULL"] = df[col].isna().astype(np.int8)

    # ── EXT_SOURCE: scores de crédito externos (features poderosas) ──────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_present = [c for c in ext_cols if c in df.columns]
    if ext_present:
        df["EXT_SOURCE_MEAN"]    = df[ext_present].mean(axis=1)
        df["EXT_SOURCE_MIN"]     = df[ext_present].min(axis=1)
        df["EXT_SOURCE_MAX"]     = df[ext_present].max(axis=1)
        df["EXT_SOURCE_STD"]     = df[ext_present].std(axis=1)
        df["EXT_SOURCE_PRODUCT"] = df[ext_present].prod(axis=1)

    # ── Encoding de categóricas ───────────────────────────────────────────────
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

app_train = limpar_application(app_train)
app_test  = limpar_application(app_test)
print(f"  application_train após limpeza: {app_train.shape}")
print(f"  application_test  após limpeza: {app_test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — FEATURE ENGINEERING NAS TABELAS PERIFÉRICAS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 3] Feature engineering nas tabelas periféricas...\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3A. BUREAU.CSV — Créditos externos (Bureau de crédito)
# ─────────────────────────────────────────────────────────────────────────────
print("  [3A] bureau.csv...")

bureau["FLAG_AMT_ANNUITY_NULL"]    = bureau["AMT_ANNUITY"].isna().astype(np.int8)
bureau["FLAG_MAX_OVERDUE_NULL"]    = bureau["AMT_CREDIT_MAX_OVERDUE"].isna().astype(np.int8)
bureau["CREDIT_ACTIVE_BIN"]        = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.int8)
bureau["DEBT_CREDIT_RATIO"]        = (
    bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1e-5)
)
bureau["OVERDUE_DEBT_RATIO"]       = (
    bureau["AMT_CREDIT_MAX_OVERDUE"] / (bureau["AMT_CREDIT_SUM_DEBT"] + 1e-5)
)

bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    bureau_n_credits             =("SK_ID_BUREAU",              "count"),
    bureau_n_active              =("CREDIT_ACTIVE_BIN",         "sum"),
    bureau_days_credit_mean      =("DAYS_CREDIT",               "mean"),
    bureau_days_credit_max       =("DAYS_CREDIT",               "max"),
    bureau_days_credit_std       =("DAYS_CREDIT",               "std"),
    bureau_overdue_days_mean     =("CREDIT_DAY_OVERDUE",        "mean"),
    bureau_overdue_days_max      =("CREDIT_DAY_OVERDUE",        "max"),
    bureau_amt_credit_sum        =("AMT_CREDIT_SUM",            "sum"),
    bureau_amt_debt_sum          =("AMT_CREDIT_SUM_DEBT",       "sum"),
    bureau_amt_overdue_mean      =("AMT_CREDIT_MAX_OVERDUE",    "mean"),
    bureau_debt_ratio_mean       =("DEBT_CREDIT_RATIO",         "mean"),
    bureau_overdue_ratio_mean    =("OVERDUE_DEBT_RATIO",        "mean"),
    bureau_flag_annuity_null_sum =("FLAG_AMT_ANNUITY_NULL",     "sum"),
    bureau_flag_overdue_null_sum =("FLAG_MAX_OVERDUE_NULL",     "sum"),
).reset_index()

bureau_agg["bureau_active_ratio"] = (
    bureau_agg["bureau_n_active"] / (bureau_agg["bureau_n_credits"] + 1e-5)
)


# ─────────────────────────────────────────────────────────────────────────────
# 3B. BUREAU_BALANCE.CSV — Histórico mensal de créditos externos
# ANTI DATA-LEAKAGE: apenas MONTHS_BALANCE < 0 (dados históricos)
# ─────────────────────────────────────────────────────────────────────────────
print("  [3B] bureau_balance.csv (com janelas temporais)...")

STATUS_MAP = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
bb["STATUS_NUM"] = bb["STATUS"].map(STATUS_MAP).fillna(0).astype(np.int8)

# Apenas histórico passado (anti-leakage)
bb_hist = bb[bb["MONTHS_BALANCE"] < 0].copy()

# Merge com bureau para obter SK_ID_CURR
bb_hist = bb_hist.merge(
    bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates(),
    on="SK_ID_BUREAU", how="left"
).dropna(subset=["SK_ID_CURR"])

bb_hist["SK_ID_CURR"] = bb_hist["SK_ID_CURR"].astype(np.int32)

# Janelas temporais: últimos 3, 6, 12 meses
bb_results = []
for window, suffix in [(3, "3m"), (6, "6m"), (12, "12m")]:
    subset = bb_hist[bb_hist["MONTHS_BALANCE"] >= -window]
    agg = subset.groupby("SK_ID_CURR")["STATUS_NUM"].agg(
        **{f"bb_status_mean_{suffix}": "mean",
           f"bb_status_max_{suffix}":  "max",
           f"bb_status_std_{suffix}":  "std",
           f"bb_n_records_{suffix}":   "count"}
    ).reset_index()
    bb_results.append(agg)

from functools import reduce
bb_agg = reduce(lambda l, r: l.merge(r, on="SK_ID_CURR", how="outer"), bb_results)

del bb_hist, bb
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 3C. INSTALLMENTS_PAYMENTS.CSV — Histórico de pagamentos
# Feature CENTRAL: atraso real (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
# ─────────────────────────────────────────────────────────────────────────────
print("  [3C] installments_payments.csv (janelas 3/6/12 parcelas)...")

inst["DELAY_DAYS"]     = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["PAYMENT_RATIO"]  = inst["AMT_PAYMENT"]        / (inst["AMT_INSTALMENT"] + 1e-5)
inst["IS_LATE"]        = (inst["DELAY_DAYS"] > 0).astype(np.int8)
inst["IS_VERY_LATE"]   = (inst["DELAY_DAYS"] > 30).astype(np.int8)
inst["AMT_UNDERPAID"]  = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]).clip(lower=0)
inst["FLAG_PAY_NULL"]  = inst["AMT_PAYMENT"].isna().astype(np.int8)

inst_sorted = inst.sort_values("DAYS_INSTALMENT", ascending=False)

def agg_inst_window(df, window_n, suffix):
    subset = df.groupby("SK_ID_CURR", group_keys=False).head(window_n)
    agg = subset.groupby("SK_ID_CURR").agg(
        **{f"inst_delay_mean_{suffix}":       ("DELAY_DAYS",    "mean"),
           f"inst_delay_max_{suffix}":        ("DELAY_DAYS",    "max"),
           f"inst_delay_std_{suffix}":        ("DELAY_DAYS",    "std"),
           f"inst_pay_ratio_mean_{suffix}":   ("PAYMENT_RATIO", "mean"),
           f"inst_pay_ratio_min_{suffix}":    ("PAYMENT_RATIO", "min"),
           f"inst_pay_ratio_std_{suffix}":    ("PAYMENT_RATIO", "std"),
           f"inst_is_late_sum_{suffix}":      ("IS_LATE",       "sum"),
           f"inst_is_very_late_sum_{suffix}": ("IS_VERY_LATE",  "sum"),
           f"inst_underpaid_sum_{suffix}":    ("AMT_UNDERPAID", "sum"),
           f"inst_n_{suffix}":               ("AMT_INSTALMENT", "count")}
    ).reset_index()
    return agg

inst_3  = agg_inst_window(inst_sorted, 3,  "3m")
inst_6  = agg_inst_window(inst_sorted, 6,  "6m")
inst_12 = agg_inst_window(inst_sorted, 12, "12m")
inst_agg = inst_3.merge(inst_6, on="SK_ID_CURR", how="outer").merge(inst_12, on="SK_ID_CURR", how="outer")

# Feature de tendência: atraso piorando ou melhorando?
inst_agg["inst_delay_trend"] = inst_agg["inst_delay_mean_3m"] - inst_agg["inst_delay_mean_12m"]

del inst, inst_sorted, inst_3, inst_6, inst_12
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 3D. POS_CASH_BALANCE.CSV — Comportamento em contratos POS e empréstimos
# ─────────────────────────────────────────────────────────────────────────────
print("  [3D] POS_CASH_balance.csv (janelas 3/6/12 meses)...")

pos_sorted = pos.sort_values("MONTHS_BALANCE", ascending=False)

def agg_pos_window(df, window_n, suffix):
    subset = df.groupby("SK_ID_CURR", group_keys=False).head(window_n)
    agg = subset.groupby("SK_ID_CURR").agg(
        **{f"pos_dpd_mean_{suffix}":     ("SK_DPD",              "mean"),
           f"pos_dpd_max_{suffix}":      ("SK_DPD",              "max"),
           f"pos_dpd_sum_{suffix}":      ("SK_DPD",              "sum"),
           f"pos_dpd_def_mean_{suffix}": ("SK_DPD_DEF",          "mean"),
           f"pos_n_{suffix}":            ("MONTHS_BALANCE",      "count")}
    ).reset_index()
    return agg

pos_3  = agg_pos_window(pos_sorted, 3,  "3m")
pos_6  = agg_pos_window(pos_sorted, 6,  "6m")
pos_12 = agg_pos_window(pos_sorted, 12, "12m")
pos_agg = pos_3.merge(pos_6, on="SK_ID_CURR", how="outer").merge(pos_12, on="SK_ID_CURR", how="outer")
pos_agg["pos_dpd_trend"] = pos_agg["pos_dpd_mean_3m"] - pos_agg["pos_dpd_mean_12m"]

del pos, pos_sorted, pos_3, pos_6, pos_12
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 3E. CREDIT_CARD_BALANCE.CSV — Comportamento de cartão de crédito
# ─────────────────────────────────────────────────────────────────────────────
print("  [3E] credit_card_balance.csv (janelas 3/6/12 meses)...")

cc["UTILIZATION"]       = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-5)
cc["FLAG_PAY_NULL"]     = cc["AMT_PAYMENT_CURRENT"].isna().astype(np.int8)
cc["PAYMENT_RATIO"]     = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1e-5)
cc_sorted = cc.sort_values("MONTHS_BALANCE", ascending=False)

def agg_cc_window(df, window_n, suffix):
    subset = df.groupby("SK_ID_CURR", group_keys=False).head(window_n)
    agg = subset.groupby("SK_ID_CURR").agg(
        **{f"cc_utilization_mean_{suffix}": ("UTILIZATION",    "mean"),
           f"cc_utilization_max_{suffix}":  ("UTILIZATION",    "max"),
           f"cc_utilization_std_{suffix}":  ("UTILIZATION",    "std"),
           f"cc_dpd_mean_{suffix}":         ("SK_DPD",         "mean"),
           f"cc_dpd_max_{suffix}":          ("SK_DPD",         "max"),
           f"cc_pay_ratio_mean_{suffix}":   ("PAYMENT_RATIO",  "mean"),
           f"cc_pay_null_sum_{suffix}":     ("FLAG_PAY_NULL",  "sum")}
    ).reset_index()
    return agg

cc_3  = agg_cc_window(cc_sorted, 3,  "3m")
cc_6  = agg_cc_window(cc_sorted, 6,  "6m")
cc_12 = agg_cc_window(cc_sorted, 12, "12m")
cc_agg = cc_3.merge(cc_6, on="SK_ID_CURR", how="outer").merge(cc_12, on="SK_ID_CURR", how="outer")
cc_agg["cc_utilization_trend"] = cc_agg["cc_utilization_mean_3m"] - cc_agg["cc_utilization_mean_12m"]

del cc, cc_sorted, cc_3, cc_6, cc_12
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 3F. PREVIOUS_APPLICATION.CSV — Propostas anteriores na Home Credit
# ─────────────────────────────────────────────────────────────────────────────
print("  [3F] previous_application.csv...")

prev["FLAG_DOWN_NULL"]         = prev["AMT_DOWN_PAYMENT"].isna().astype(np.int8)
prev["CREDIT_APP_RATIO"]       = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1e-5)
prev["IS_APPROVED"]            = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
prev["IS_REFUSED"]             = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
prev["IS_CANCELED"]            = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)

prev_agg = prev.groupby("SK_ID_CURR").agg(
    prev_n_apps             =("SK_ID_PREV",         "count"),
    prev_n_approved         =("IS_APPROVED",         "sum"),
    prev_n_refused          =("IS_REFUSED",          "sum"),
    prev_n_canceled         =("IS_CANCELED",         "sum"),
    prev_amt_credit_mean    =("AMT_CREDIT",          "mean"),
    prev_amt_credit_max     =("AMT_CREDIT",          "max"),
    prev_amt_annuity_mean   =("AMT_ANNUITY",         "mean"),
    prev_credit_app_ratio   =("CREDIT_APP_RATIO",    "mean"),
    prev_days_decision_mean =("DAYS_DECISION",       "mean"),
    prev_flag_down_null_sum =("FLAG_DOWN_NULL",      "sum"),
).reset_index()

prev_agg["prev_approval_rate"] = prev_agg["prev_n_approved"] / (prev_agg["prev_n_apps"] + 1e-5)
prev_agg["prev_refusal_rate"]  = prev_agg["prev_n_refused"]  / (prev_agg["prev_n_apps"] + 1e-5)

del prev
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — JOIN SEGURO (grain preservado: 1 linha = 1 cliente)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 4] Unindo todas as tabelas (left join por SK_ID_CURR)...\n")

tabelas_perifericas = [
    ("bureau_agg",   bureau_agg),
    ("bb_agg",       bb_agg),
    ("prev_agg",     prev_agg),
    ("inst_agg",     inst_agg),
    ("pos_agg",      pos_agg),
    ("cc_agg",       cc_agg),
]

for nome, tbl in tabelas_perifericas:
    antes = app_train.shape[0]
    app_train = app_train.merge(tbl, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(tbl,  on="SK_ID_CURR", how="left")
    depois = app_train.shape[0]
    assert antes == depois, f"ERRO: {nome} duplicou linhas! ({antes} → {depois})"
    print(f"  ✅ {nome:15s} joined | features adicionadas: +{len(tbl.columns)-1:3d} | treino shape: {app_train.shape}")

del bureau_agg, bb_agg, prev_agg, inst_agg, pos_agg, cc_agg
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — MODELAGEM COM LIGHTGBM + STRATIFIED K-FOLD
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 5] Modelagem — LightGBM com StratifiedKFold (k={N_SPLITS})...\n")

TARGET_COL = "TARGET"
IGNORE     = ["SK_ID_CURR", TARGET_COL]
FEATURES   = [c for c in app_train.columns if c not in IGNORE]

X           = app_train[FEATURES]
y           = app_train[TARGET_COL].astype(np.int8)
X_test_sub  = app_test[FEATURES].reindex(columns=FEATURES, fill_value=np.nan)
test_ids    = app_test["SK_ID_CURR"]

print(f"  Total de features : {len(FEATURES)}")
print(f"  Shape treino      : {X.shape}")
print(f"  Shape teste       : {X_test_sub.shape}")
print()

lgb_params = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "class_weight":     "balanced",  # ← trata desbalanceamento 1:11
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1,
}

cv          = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds   = np.zeros(len(X))
test_preds  = np.zeros(len(X_test_sub))
fold_aucs   = []
last_model  = None

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds        += model.predict_proba(X_test_sub)[:, 1] / N_SPLITS

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_aucs.append(fold_auc)
    last_model = model
    print(f"  Fold {fold}/{N_SPLITS} → AUC: {fold_auc:.4f}  |  best_iter: {model.best_iteration_}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n  {'─'*50}")
print(f"  AUC-ROC OOF Geral : {overall_auc:.4f}")
print(f"  AUC médio ± std   : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — ANÁLISE FINANCEIRA: PROFIT CURVE + THRESHOLD ÓTIMO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 6] Análise de impacto financeiro por threshold...\n")

amt_credit  = app_train["AMT_CREDIT"].values.astype(np.float64)
amt_annuity = app_train["AMT_ANNUITY"].values.astype(np.float64)
y_true      = y.values

thresholds = np.linspace(0.01, 0.99, 300)
results = []

for thr in thresholds:
    y_pred = (oof_preds >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    # FP: rejeitamos bom cliente → perdemos AMT_ANNUITY × prazo médio
    custo_fp = (amt_annuity[fp_mask] * PRAZO_MEDIO_MESES).sum()
    # FN: aprovamos mau cliente → perdemos (1 - recuperação) × AMT_CREDIT
    custo_fn = (amt_credit[fn_mask] * (1 - TAXA_RECUPERACAO)).sum()

    prejuizo  = custo_fp + custo_fn
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    results.append({
        "threshold": thr, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "custo_fp": custo_fp, "custo_fn": custo_fn,
        "prejuizo_total": prejuizo,
        "precision": precision, "recall": recall, "f1": f1,
    })

df_res = pd.DataFrame(results)
best   = df_res.loc[df_res["prejuizo_total"].idxmin()]

print(f"  ╔══════════════════════════════════════════════╗")
print(f"  ║       RESULTADO DO THRESHOLD ÓTIMO          ║")
print(f"  ╠══════════════════════════════════════════════╣")
print(f"  ║  Threshold              : {best['threshold']:.3f}               ║")
print(f"  ║  Custo FP (R$)          : {best['custo_fp']/1e6:>10.1f} M               ║")
print(f"  ║  Custo FN (R$)          : {best['custo_fn']/1e6:>10.1f} M               ║")
print(f"  ║  Prejuízo Total (R$)    : {best['prejuizo_total']/1e6:>10.1f} M               ║")
print(f"  ║  Precision              : {best['precision']:.4f}               ║")
print(f"  ║  Recall                 : {best['recall']:.4f}               ║")
print(f"  ║  F1-Score               : {best['f1']:.4f}               ║")
print(f"  ║  TP={best['tp']:.0f}  FP={best['fp']:.0f}  TN={best['tn']:.0f}  FN={best['fn']:.0f}  ║")
print(f"  ╚══════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — DASHBOARD DE VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[BLOCO 7] Gerando dashboard de visualizações...\n")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
ACCENT1  = "#58a6ff"   # azul
ACCENT2  = "#f78166"   # vermelho
ACCENT3  = "#56d364"   # verde
ACCENT4  = "#e3b341"   # amarelo
TEXT_CLR = "#c9d1d9"

def style_ax(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    if title:
        ax.set_title(title, color=TEXT_CLR, fontweight="bold", fontsize=10, pad=8)
    ax.grid(alpha=0.15, color="#30363d", linestyle="--")

# ── 1. Profit Curve ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Profit Curve — Custo Financeiro por Threshold")
ax1.plot(df_res["threshold"], df_res["prejuizo_total"] / 1e6,
         color=ACCENT2, lw=2.5, label="Prejuízo Total (FP + FN)")
ax1.plot(df_res["threshold"], df_res["custo_fp"] / 1e6,
         color=ACCENT1, lw=1.5, ls="--", alpha=0.8, label="Custo FP (Bom cliente rejeitado)")
ax1.plot(df_res["threshold"], df_res["custo_fn"] / 1e6,
         color=ACCENT4, lw=1.5, ls="--", alpha=0.8, label="Custo FN (Mau cliente aprovado)")
ax1.axvline(best["threshold"], color=ACCENT3, lw=2, ls=":",
            label=f"Threshold Ótimo = {best['threshold']:.3f}")
ax1.set_xlabel("Threshold de Probabilidade", color=TEXT_CLR)
ax1.set_ylabel("Custo (R$ Milhões)", color=TEXT_CLR)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:.0f}M"))
leg = ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#30363d", labelcolor=TEXT_CLR)

# ── 2. Distribuição TARGET ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "Distribuição da Variável TARGET")
counts = [vc[0], vc[1]]
bars   = ax2.bar(["Adimplente (0)", "Inadimplente (1)"], counts,
                  color=[ACCENT3, ACCENT2], width=0.5, edgecolor="#30363d", lw=0.8)
for bar, cnt in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f"{cnt:,}\n({cnt/len(app_train)*100:.1f}%)",
             ha="center", color=TEXT_CLR, fontsize=9, fontweight="bold")
ax2.set_ylabel("Quantidade", color=TEXT_CLR)
ax2.set_ylim(0, max(counts) * 1.15)

# ── 3. Precision × Recall × F1 ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3, "Precision / Recall / F1 por Threshold")
ax3.plot(df_res["threshold"], df_res["precision"], color=ACCENT1, lw=2,   label="Precision")
ax3.plot(df_res["threshold"], df_res["recall"],    color=ACCENT4, lw=2,   label="Recall")
ax3.plot(df_res["threshold"], df_res["f1"],        color=ACCENT3, lw=2,   label="F1-Score")
ax3.axvline(best["threshold"], color=ACCENT2, lw=1.5, ls=":",
            label=f"Threshold Ótimo = {best['threshold']:.3f}")
ax3.set_xlabel("Threshold", color=TEXT_CLR)
ax3.set_ylabel("Score", color=TEXT_CLR)
ax3.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#30363d", labelcolor=TEXT_CLR)
ax3.set_ylim(0, 1.05)

# ── 4. Matriz de Confusão ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, f"Matriz de Confusão\n(Threshold = {best['threshold']:.3f})")
cm_vals = np.array([[int(best["tn"]), int(best["fp"])],
                    [int(best["fn"]), int(best["tp"])]])
cm_labels = [["TN\n(Adimplente\ncorreto)", "FP\n(Bom cliente\nrejeitado)"],
             ["FN\n(Inadimplente\naprovado)", "TP\n(Inadimplente\nbloqueado)"]]
cm_colors = [[ACCENT3, ACCENT2], [ACCENT4, ACCENT1]]
for i in range(2):
    for j in range(2):
        ax4.add_patch(plt.Rectangle((j-0.5, 1.5-i), 1, 1,
                                     color=cm_colors[i][j], alpha=0.3))
        ax4.text(j, 1-i, f"{cm_vals[i,j]:,}\n{cm_labels[i][j]}",
                 ha="center", va="center", fontsize=9,
                 color=TEXT_CLR, fontweight="bold")
ax4.set_xlim(-0.5, 1.5); ax4.set_ylim(-0.5, 1.5)
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Predito 0", "Predito 1"], color=TEXT_CLR)
ax4.set_yticklabels(["Real 1", "Real 0"], color=TEXT_CLR)
ax4.grid(False)

# ── 5. Feature Importance Top 25 ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
style_ax(ax5, "Top 25 Features por Importância — LightGBM (último fold)")

feat_imp = pd.Series(last_model.feature_importances_, index=FEATURES)
top25    = feat_imp.nlargest(25).sort_values()

def get_color(fname):
    if any(k in fname for k in ["inst_", "pos_", "cc_", "bureau", "bb_", "prev_"]):
        return ACCENT2   # tabelas periféricas = vermelho
    elif "EXT_SOURCE" in fname:
        return ACCENT3   # score externo = verde
    elif "AMT_" in fname or "CREDIT_" in fname or "ANNUITY" in fname:
        return ACCENT4   # financeiro = amarelo
    return ACCENT1       # outros = azul

colors = [get_color(f) for f in top25.index]
ax5.barh(top25.index, top25.values, color=colors, edgecolor="#30363d", lw=0.5, height=0.75)
ax5.set_xlabel("Importância (ganho)", color=TEXT_CLR)
ax5.tick_params(axis="y", labelsize=8)

legend_patches = [
    Patch(color=ACCENT2, label="Tabelas Periféricas (Comportamental)"),
    Patch(color=ACCENT3, label="Score Externo (EXT_SOURCE)"),
    Patch(color=ACCENT4, label="Variáveis Financeiras"),
    Patch(color=ACCENT1, label="Outras Variáveis Cadastrais"),
]
ax5.legend(handles=legend_patches, fontsize=8, facecolor=CARD_BG,
           edgecolor="#30363d", labelcolor=TEXT_CLR, loc="lower right")

# ── Título global ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.98,
         "Squad 3 — Motor de Risco Transacional | Home Credit Default Risk",
         ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT_CLR)
fig.text(0.5, 0.96,
         f"AUC-ROC OOF: {overall_auc:.4f}  |  Threshold Ótimo: {best['threshold']:.3f}  |  "
         f"Premissas: Taxa Recuperação={TAXA_RECUPERACAO*100:.0f}%  Prazo={PRAZO_MEDIO_MESES}m",
         ha="center", va="top", fontsize=9, color="#8b949e")

out_img = os.path.join(OUTPUT_DIR, "squad3_dashboard.png")
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ Dashboard salvo em: {out_img}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 8 — SUBMISSION + RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_preds})
submission.to_csv(os.path.join(OUTPUT_DIR, "squad3_submission.csv"), index=False)
print(f"  ✅ Submission salva  : {OUTPUT_DIR}\\squad3_submission.csv")

# Relatório de features geradas
feat_report = feat_imp.sort_values(ascending=False).reset_index()
feat_report.columns = ["feature", "importance"]
feat_report["source"] = feat_report["feature"].apply(
    lambda f: "Periférica" if any(k in f for k in ["inst_","pos_","cc_","bureau","bb_","prev_"])
              else "Principal"
)
feat_report.to_csv(os.path.join(OUTPUT_DIR, "squad3_feature_importance.csv"), index=False)
print(f"  ✅ Feature report    : {OUTPUT_DIR}\\squad3_feature_importance.csv")

print(f"\n{DIVIDER}")
print(f"  PIPELINE CONCLUÍDO")
print(f"  AUC-ROC OOF  : {overall_auc:.4f}")
print(f"  Total features: {len(FEATURES)}")
print(f"  Threshold ótimo (negócio): {best['threshold']:.3f}")
print(f"  Custo financeiro mínimo  : R$ {best['prejuizo_total']/1e6:.1f}M")
print(DIVIDER)
