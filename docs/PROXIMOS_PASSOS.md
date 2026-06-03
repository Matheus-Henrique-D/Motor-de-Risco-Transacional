# 📋 Próximos Passos e Roteiro de Finalização — Squad 3

Este documento detalha o status dos entregáveis do desafio e fornece um roteiro estruturado para a escrita do relatório executivo e preparação da apresentação final (Pitch).

---

## 🚦 Status dos Entregáveis (Squad 3)

| Entregável | Requisito | Status | Descrição / Observações |
| :--- | :--- | :--- | :--- |
| **Entregável 1** | Aplicação Interativa Streamlit | **🟢 Concluído** | Desenvolvido em `app.py`. O upload de CSV está operando em alta performance e as decisões (**Aprovar/Revisar/Bloquear**) estão calibradas com o threshold ótimo do modelo. |
| **Entregável 1** | Modelo `.pkl` Embarcado | **🟢 Concluído** | O pipeline gera automaticamente o arquivo `output/squad3_lgb_ensemble.pkl` com os 5 classificadores do ensemble e os Label Encoders categóricos. |
| **Entregável 2** | Documentação Executiva (PDF) | **🔴 Pendente** | É necessário estruturar e redigir o relatório sob o framework SEMMA e exportá-lo para formato PDF. |
| **Pitch** | Apresentação de 15 Minutos | **🔴 Pendente** | Montar a estrutura de slides e definir os papéis de apresentação da equipe (Board Meeting). |

---

## 📄 Roteiro e Estrutura: Relatório Executivo (SEMMA)

O relatório deve focar em traduzir os aspectos técnicos de Machine Learning em **valor de negócio** para o cliente (*Home Credit Finanças*). Use a estrutura abaixo para redigir o documento:

### 1. Sumário Executivo
- **Problema de Negócio**: Concessão de crédito para clientes desbancarizados (sem histórico tradicional no Serasa/SPC) usando dados transacionais históricos e comportamento transacional periférico.
- **A Solução**: Um motor preditivo que calcula o risco e recomenda a melhor decisão comercial (aprovação, análise manual ou bloqueio) de forma a maximizar o lucro operacional do banco.

### 2. A Jornada (SEMMA)
- **Sample & Explore**:
  - *Desbalanceamento*: Explicação sobre a baixa taxa de inadimplência histórica (~8% de inadimplentes na base cadastral).
  - *Validação*: Uso de validação cruzada estratificada em 5 folds (`StratifiedKFold`) para blindar o modelo contra *overfitting*.
  - *Tratamento de Anomalias*: Substituição do código anômalo `365243` na variável `DAYS_EMPLOYED` por dados nulos estruturais (referente a aposentados/não empregados).
- **Modify (As 3 Melhores Features Criadas)**:
  1. `ANNUITY_CREDIT_RATIO` (Tabela Principal): Razão entre o valor da anuidade e o crédito total solicitado. Excelente indicador do custo do empréstimo e do prazo implícito da operação.
  2. `bureau_debt_ratio_mean` (Tabela Bureau): Comprometimento financeiro médio do cliente junto a outros bancos externos.
  3. `inst_delay_trend` (Tabela Parcelas): Tendência de atrasos de pagamento comparando o curto prazo (média dos últimos 3 meses) com o longo prazo (últimos 12 meses).
- **Model**:
  - O algoritmo vencedor foi o **LightGBM Classifier** (Gradient Boosting), parametrizado com `class_weight='balanced'` para lidar com a assimetria das classes.
  - A performance de treino estabilizou na métrica **AUC-ROC OOF Geral de `0.7843`**.
- **Assess (O Impacto Financeiro e Threshold)**:
  - *O Threshold Ótimo*: **`0.931` (93.1%)**.
  - *Justificativa de Negócio*: No mercado de crédito B2C desbancarizado, o custo de oportunidade de rejeitar um bom cliente (Falso Positivo) que traria 24 meses de juros e capital é extremamente alto em relação ao risco de aceitar um cliente inadimplente. Por isso, a Curva de Lucro (*Profit Curve*) calibrou o corte ideal de risco em 93.1%, aprovando a grande maioria (94.8% da base) e enviando apenas os perfis cinzentos (5.2%) para análise humana na mesa de crédito.

---

## 🎤 Roteiro e Slides Recomendados para o Pitch (15 minutos)

A banca avaliadora atuará como a Diretoria Executiva da *Home Credit*. A apresentação deve ter foco comercial:

* **Slide 1: Capa e Introdução** (Squad 3 - Consultoria de Data Science).
* **Slide 2: A Oportunidade** (Desbancarizados: Como expandir a base de clientes do banco com segurança, sem dispor de Serasa/SPC).
* **Slide 3: Engenharia de Atributos (SEMMA - Modify)** (Explicar de forma simples como as variáveis de bureaus e parcelas traduzem o caráter de pagamento do cliente).
* **Slide 4: A Racionalidade Financeira (Assess - O Clímax)** (Apresentar o gráfico da *Profit Curve* de `squad3_dashboard.png`. Explicar por que aprovar a maioria dos clientes é a decisão que traz mais lucro para o banco).
* **Slide 5: Demonstração ao vivo do App Streamlit** (Faça o upload do arquivo de amostra de 100 clientes na tela para mostrar a velocidade e usabilidade do sistema preditivo).
* **Slide 6: Conclusão e Próximos Passos** (Agradecimento e abertura para perguntas da diretoria).
