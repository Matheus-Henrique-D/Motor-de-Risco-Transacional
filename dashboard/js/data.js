/**
 * data.js — Fonte de dados do pipeline Squad 3
 * Dados reais extraídos da execução do squad3_pipeline.py
 * Substitua os valores caso rode o pipeline novamente.
 */
export const pipelineData = {
  model: {
    auc_roc:        0.7843,
    threshold:      0.931,
    total_features: 249,
    n_splits:       5,
    algorithm:      "LightGBM",
    class_weight:   "balanced",
  },

  confusion_matrix: {
    tn: 282673,
    fp: 13,
    fn: 24764,
    tp: 61,
    threshold: 0.931,
  },

  target_distribution: {
    adimplente:   { count: 282686, label: "Adimplente (0)" },
    inadimplente: { count: 24825,  label: "Inadimplente (1)" },
    total:        307511,
  },

  business_assumptions: [
    {
      icon: "💰",
      label: "Taxa de Recuperação",
      value: "20%",
      desc: "Percentual do crédito recuperado em caso de inadimplência confirmada.",
      color: "var(--accent-yellow)",
    },
    {
      icon: "📅",
      label: "Prazo Médio",
      value: "24 meses",
      desc: "Proxy de prazo para cálculo de receita de anuidade no caso de FP.",
      color: "var(--accent-blue)",
    },
    {
      icon: "❌",
      label: "Custo do FP",
      value: "AMT_ANNUITY × 24",
      desc: "Bom cliente rejeitado: perda de receita futura de juros.",
      color: "var(--accent-red)",
    },
    {
      icon: "⚠️",
      label: "Custo do FN",
      value: "AMT_CREDIT × 80%",
      desc: "Mau cliente aprovado: perda líquida do capital não recuperado.",
      color: "var(--accent-orange)",
    },
  ],

  summary_cards: [
    { label: "AUC-ROC OOF",       value: "0.7843",   sub: "Média nos 5 folds",          color: "value--green"  },
    { label: "Threshold Ótimo",   value: "0.931",    sub: "Mínimo prejuízo financeiro", color: "value--blue"   },
    { label: "Total de Features", value: "249",      sub: "Após feature engineering",   color: "value--purple" },
    { label: "Registros Treino",  value: "307.511",  sub: "application_train.csv",      color: "value--yellow" },
    { label: "Registros Teste",   value: "48.744",   sub: "application_test.csv",       color: "value--orange" },
    { label: "Folds CV",          value: "5",        sub: "StratifiedKFold",            color: "value--red"    },
  ],

  dataset_overview: [
    {
      file:    "application_train.csv",
      rows:    307511,
      cols:    122,
      nullPct: 24.40,
      key:     "SK_ID_CURR",
      desc:    "Tabela central de treino. Contém TARGET (variável alvo).",
    },
    {
      file:    "application_test.csv",
      rows:    48744,
      cols:    121,
      nullPct: 23.81,
      key:     "SK_ID_CURR",
      desc:    "Tabela central de teste. Sem TARGET.",
    },
    {
      file:    "bureau.csv",
      rows:    1716428,
      cols:    17,
      nullPct: 13.50,
      key:     "SK_ID_BUREAU",
      desc:    "Créditos externos em bureaus (Serasa/SPC equivalente).",
    },
    {
      file:    "bureau_balance.csv",
      rows:    27299925,
      cols:    3,
      nullPct: 0.00,
      key:     "SK_ID_BUREAU",
      desc:    "Histórico mensal de status de cada crédito externo.",
    },
    {
      file:    "previous_application.csv",
      rows:    1670214,
      cols:    37,
      nullPct: 17.98,
      key:     "SK_ID_PREV",
      desc:    "Todas as propostas anteriores na própria Home Credit.",
    },
    {
      file:    "POS_CASH_balance.csv",
      rows:    10001358,
      cols:    8,
      nullPct: 0.07,
      key:     "SK_ID_PREV",
      desc:    "Saldo mensal de empréstimos POS e crédito pessoal.",
    },
    {
      file:    "installments_payments.csv",
      rows:    13605401,
      cols:    8,
      nullPct: 0.01,
      key:     "SK_ID_PREV",
      desc:    "Histórico de parcelas geradas vs. pagamentos efetuados.",
    },
    {
      file:    "credit_card_balance.csv",
      rows:    3840312,
      cols:    23,
      nullPct: 6.65,
      key:     "SK_ID_PREV",
      desc:    "Saldo mensal e comportamento de cartões de crédito.",
    },
  ],

  // Top 25 features do squad3_feature_importance.csv (dados reais)
  feature_importance: [
    { feature: "ANNUITY_CREDIT_RATIO",   importance: 1291, source: "Principal"  },
    { feature: "AGE_YEARS",              importance: 897,  source: "Principal"  },
    { feature: "EXT_SOURCE_3",           importance: 840,  source: "Principal"  },
    { feature: "prev_amt_annuity_mean",  importance: 825,  source: "Periférica" },
    { feature: "prev_days_decision_mean",importance: 782,  source: "Periférica" },
    { feature: "EXT_SOURCE_1",           importance: 775,  source: "Principal"  },
    { feature: "EXT_SOURCE_MEAN",        importance: 771,  source: "Principal"  },
    { feature: "DAYS_EMPLOYED",          importance: 741,  source: "Principal"  },
    { feature: "bureau_debt_ratio_mean", importance: 710,  source: "Periférica" },
    { feature: "prev_amt_credit_max",    importance: 669,  source: "Periférica" },
    { feature: "YEARS_ID_PUBLISH",       importance: 668,  source: "Principal"  },
    { feature: "bureau_days_credit_max", importance: 656,  source: "Periférica" },
    { feature: "bureau_amt_credit_sum",  importance: 623,  source: "Periférica" },
    { feature: "EXT_SOURCE_STD",         importance: 615,  source: "Principal"  },
    { feature: "prev_credit_app_ratio",  importance: 608,  source: "Periférica" },
    { feature: "AMT_ANNUITY",            importance: 594,  source: "Principal"  },
    { feature: "EXT_SOURCE_2",           importance: 584,  source: "Principal"  },
    { feature: "EXT_SOURCE_PRODUCT",     importance: 582,  source: "Principal"  },
    { feature: "YEARS_REGISTRATION",     importance: 574,  source: "Principal"  },
    { feature: "bureau_days_credit_mean",importance: 570,  source: "Periférica" },
    { feature: "ANNUITY_INCOME_RATIO",   importance: 550,  source: "Principal"  },
    { feature: "bureau_days_credit_std", importance: 543,  source: "Periférica" },
    { feature: "prev_amt_credit_mean",   importance: 538,  source: "Periférica" },
    { feature: "EXT_SOURCE_MAX",         importance: 531,  source: "Principal"  },
    { feature: "inst_delay_trend",       importance: 528,  source: "Periférica" },
  ],
};
