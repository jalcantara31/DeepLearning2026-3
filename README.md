# DeepLearning2026-3: Monitoramento Baseado em Condição (CBM) e Predição de Séries Temporais em Compressores Industriais

Este repositório contém um ecossistema completo para **Análise Exploratória de Dados (EDA)** e um **Pipeline Modular de Deep Learning** para modelagem e previsão de séries temporais de sensores industriais provenientes de um compressor centrífugo (`data/compressor.csv`).

O projeto implementa arquiteturas neurais recorrentes (**LSTM** e **GRU**) em **PyTorch puro**, com foco rigoroso em **zero *data leakage***, validação termodinâmica e aceleração por GPU (**CUDA / NVIDIA RTX 3070 Ti**).

---

## Sumário

1. [Visão Geral do Problema e Dados](#1-visão-geral-do-problema-e-dados)
2. [Fundamentação Teórica](#2-fundamentação-teórica)
   - [Termodinâmica de Compressão Politrópica](#21-termodinâmica-de-compressão-politrópica)
   - [Amostragem em Historiadores Industriais e o Fenômeno Split-Row](#22-amostragem-em-historiadores-industriais-e-o-fenômeno-split-row)
   - [Prevenção Rigorosa de Data Leakage em Séries Temporais](#23-prevenção-rigorosa-de-data-leakage-em-séries-temporais)
   - [Modelos Recorrentes: LSTM vs. GRU](#24-modelos-recorrentes-lstm-vs-gru)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Módulos do Pipeline](#4-módulos-do-pipeline)
5. [Instalação e Configuração do Ambiente](#5-instalação-e-configuração-do-ambiente)
6. [Como Executar](#6-como-executar)
   - [Treinamento e Avaliação (CLI)](#61-treinamento-e-avaliação-cli)
   - [Configuração via YAML](#62-configuração-via-yaml)
7. [Métricas de Desempenho e Avaliação](#7-métricas-de-desempenho-e-avaliação)
8. [Resultados e Artefatos Gerados](#8-resultados-e-artefatos-gerados)

---

## 1. Visão Geral do Problema e Dados

O arquivo [`data/compressor.csv`](data/compressor.csv) contém leituras coletadas entre **01/09/2019 e 01/03/2020** (182 dias de operação contínua, totalizando **547.508 registros**) referentes ao **1º Estágio de Compressão (CBM — *Condition-Based Monitoring*)**.

As variáveis monitoradas dividem-se em 4 famílias termodinâmicas, cada uma contendo três grandezas: **Actual** (medida real instantânea), **Expected** (curva teórica/projeto) e **Deviation** (desvio percentual relativo):

| Família Física | Série *Actual* | Série *Expected* | Série *Deviation* | Unidade / Descrição |
| :--- | :--- | :--- | :--- | :--- |
| **Razão de Pressão ($r_p$)** | `act_press_ratio` | `exp_press_ratio` | `press_rat_dev` | Razão $P_{\text{descarga}} / P_{\text{sucção}}$ |
| **Head Politrópico ($H_p$)** | `act_poly_head` | `exp_poly_head` | `poly_head_dev` | Energia específica transferida ao fluido ($\text{kJ/kg}$) |
| **Potência de Eixo ($W_s$)** | `act_shaft_power` | `exp_shaft_power` | `shaft_pwr_dev` | Potência consumida no eixo ($\text{kW}$) |
| **Eficiência Politrópica ($\eta_p$)** | `act_poly_eff` | `exp_poly_eff` | `poly_eff_dev` | Rendimento termodinâmico da compressão ($\%$) |

---

## 2. Fundamentação Teórica

### 2.1 Termodinâmica de Compressão Politrópica

Em compressores centrífugos de múltiplos estágios, a compressão do gás é modelada pelo processo politrópico:

$$P \cdot v^n = \text{constante}$$

Onde $n$ é o expoente politrópico. A relação entre a razão de compressão $r_p = \frac{P_2}{P_1}$ e o Head Politrópico $H_p$ é dada por:

$$H_p = \frac{Z_{\text{med}} \cdot R \cdot T_1}{\frac{n-1}{n}} \left[ r_p^{\frac{n-1}{n}} - 1 \right]$$

A eficiência politrópica $\eta_p$ relaciona o trabalho politrópico útil e o consumo real no eixo ($W_s$):

$$\eta_p = \frac{\dot{m} \cdot H_p}{W_s}$$

#### Equação de Desvio CBM
O sistema de monitoramento de integridade baseia-se no desvio relativo entre o ponto de operação medido e o esperado pelo modelo de projeto:

$$\text{Dev (\%)} = \frac{\text{Actual} - \text{Expected}}{\text{Expected}} \times 100$$

> **Reconstrução Analítica:** No arquivo bruto, a coluna `PH (CBM) 1st Stage Poly Eff Dev` apresentava-se 100% nula. Como comprovado pela auditoria em [`eda/analise_compressor.ipynb`](eda/analise_compressor.ipynb), a identidade dos desvios é exata ($R^2 > 0{,}9999$). O pipeline reconstrói de forma determinística e instantânea essa variável, recuperando o indicador para a modelagem preditiva.

---

### 2.2 Amostragem em Historiadores Industriais e o Fenômeno *Split-Row*

Historiadores de processos contínuos (como OSIsoft PI / AVEVA) utilizam mecanismos de compressão e protocolos de varredura assíncronos. Dois fenômenos cruciais foram identificados nos dados:

1. **Assincronia de Varredura (*Split-Row Clock Skew*):** Grupos de tags com rotinas de cálculo distintas chegam com 1 a 2 segundos de diferença (`28s + 2s = 30s`). Isso gera registros incompletos alternados com aparência de falha de sensor (~5,24% de NaNs). Ao aplicar uma agregação em grade temporal regular (ex.: `5min`), pares complementares são fundidos sem qualquer perda de informação, elevando a completude para **>99,8%**.
2. **Compressão por Zona Morta (*Deadband / Exception Compression*):** Em paradas de máquina, quando as variáveis permanecem em zero, o historiador estende o intervalo de escrita para 10 minutos (`600s`), evitando redundância de gravação de estados estáticos.

---

### 2.3 Prevenção Rigorosa de *Data Leakage* em Séries Temporais

*Data Leakage* (vazamento de dados) é a causa mais comum de métricas excessivamente otimistas que falham em produção. O pipeline aplica garantias arquiteturais rígidas:

```
Linha do Tempo (01/Set/2019 a 01/Mar/2020)
├────────────────────────────┼─GAP─┼──────────────┼─GAP─┼──────────────┤
│       TREINO (70%)         │     │ VALIDAÇÃO (15%)    │  TESTE (15%) │
│   (Fit dos Scalers Aqui)   │     │ (Transform Only)   │(Transform Only)
└────────────────────────────┴─────┴──────────────┴─────┴──────────────┘
```

1. **Separação Cronológica Precoce:** A divisão em Treino, Validação e Teste é feita **antes** de qualquer imputação ou normalização. Jamais se utiliza *k-fold* aleatório ou *shuffling* em dados temporais contínuos.
2. **Isolamento de Scalers:** Os objetos `StandardScaler` (ou `RobustScaler` / `MinMaxScaler`) executam `.fit()` **exclusivamente nas linhas válidas da partição de Treino**. As partições de Validação e Teste são apenas transformadas (`.transform()`).
3. **Imputação Causal Estrita:** O preenchimento de micro-lacunas utiliza apenas *forward-fill* (`ffill`) isolado dentro de cada partição, garantindo que leituras em $t$ não recebam informações do futuro ($t+1$).
4. **Embargo Temporal (*Split Gap*):** Uma barreira configurável descarta as transições imediatas entre o fim do treino e início da validação, prevenindo autocorrelação residual entre as fronteiras.
5. **Auditoria em Runtime:** O método `TimeSeriesPreprocessor.verify_no_leakage` executa asserções formais em tempo de execução garantindo monotonicidade temporal, interseção nula de timestamps e conformidade dos parâmetros estatísticos do scaler.

---

### 2.4 Modelos Recorrentes: LSTM vs. GRU

Modelos recorrentes foram concebidos para capturar dependências e memórias em sequências temporais através de estados ocultos recorrentes:

```
Entrada:  X = [x_{t-L}, x_{t-L+1}, ..., x_{t-1}]  (L = lookback)
Saída:    y_hat = [y_t, ..., y_{t+H-1}]           (H = horizon)
```

#### LSTM (Long Short-Term Memory)
A LSTM mitiga o problema do desvanecimento do gradiente (*vanishing gradient*) mantendo uma célula de estado constante $c_t$ controlada por 3 portas (gates):

$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad &&\text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad &&\text{(Input Gate)} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad &&\text{(Candidate State)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &&\text{(Cell State)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(c_t) \quad &&\text{(Hidden State)}
\end{aligned}$$

#### GRU (Gated Recurrent Unit)
A GRU simplifica a dinâmica unificando o estado de célula e o estado oculto, operando com 2 portas:

$$\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad &&\text{(Update Gate)} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad &&\text{(Reset Gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \quad &&\text{(Candidate Hidden State)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &&\text{(Hidden State)}
\end{aligned}$$

- **Vantagem da GRU:** Menor número de parâmetros (~25% menos pesos em relação à LSTM com mesmo `hidden_size`), convergência mais rápida e menor risco de sobreajuste em regimes de dados moderados.
- **Vantagem da LSTM:** Maior expressividade para modelar dinâmicas de longo prazo que demandam separação explícita entre memória de longo prazo ($c_t$) e foco de curto prazo ($h_t$).

---

## 3. Estrutura do Repositório

```
DeepLearning2026-3/
├── README.md                      # Documentação completa e fundamentação teórica
├── requirements.txt               # Dependências do projeto (PyTorch, Pandas, Scikit-Learn, etc.)
├── run.py                         # Entrypoint CLI na raiz
├── scripts/
│   └── run_eda.py                 # Script autônomo para rodar a EDA e exportar os plots em PDF
├── data/
│   └── compressor.csv             # Conjunto de dados multivariado (547k linhas)
├── eda/
│   └── analise_compressor.ipynb   # Notebook com EDA e diagnósticos termodinâmicos
├── ai/
│   ├── __init__.py
│   ├── config.yaml                # Configuração padrão (treina e compara GRU e LSTM)
│   ├── run.py                     # Entrypoint CLI dentro do pacote ai
│   ├── pipeline.py                # Wrapper para importação do orquestrador
│   ├── configs/                   # Perfis específicos em YAML
│   │   ├── config.yaml            # Configuração unificada padrão
│   │   ├── gru.yaml               # Treinamento focado em GRU
│   │   └── lstm.yaml              # Treinamento focado em LSTM
│   ├── loader/
│   │   ├── __init__.py
│   │   └── data_loader.py         # Leitura, parsing temporal e reconstrução de tags CBM
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── preprocessor.py        # Pipeline anti-leakage e regularização temporal
│   │   └── dataset.py             # SlidingWindowDataset e DataLoaders PyTorch
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Classe abstrata BaseTimeSeriesForecaster
│   │   ├── gru.py                 # Arquitetura GRUForecaster
│   │   ├── lstm.py                # Arquitetura LSTMForecaster
│   │   └── factory.py             # Factory de instanciação de modelos por chave
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Engine de treino, validação, EarlyStopping e checkpoints
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py           # Cálculo de métricas físicas, plots e relatórios
└── results/
    └── ai/                        # Checkpoints (.pt), curvas (.png), previsões (.csv) e JSONs
```

---

## 4. Módulos do Pipeline

### 4.1 Carregador de Dados (`ai.loader.data_loader`)
- Gerencia caminhos relativos/absolutos de modo seguro.
- Ordena o índice cronológico e consolida leituras simultâneas por média instantânea.
- Remove colunas estruturalmente vazias e calcula analiticamente `poly_eff_dev` de forma estritamente univariada no tempo $t$.

### 4.2 Pré-processamento e Divisão (`ai.preprocessing.preprocessor`)
- **Regularização Temporal:** Converte os dados brutos de taxa irregular para uma grade regular (padrão: `5min`).
- **Filtro Físico Instantâneo:** Isola paradas operacionais ($W_s \le 4000\text{ kW}$) e picos espúrios de partida ($\eta_p < 50\%$ ou $\eta_p > 100\%$).
- **Ajuste Confinado:** Realiza o `.fit()` dos scalers apenas no treino e propaga a transformação para validação e teste.

### 4.3 Criação de Janelas (`ai.preprocessing.dataset`)
- [`SlidingWindowDataset`](ai/preprocessing/dataset.py): Implementação otimizada via somas cumulativas que localiza e extrai em $O(1)$ apenas janelas temporais contínuas e sem NaNs.
- Validação estrutural de que nenhuma janela de treino invade o tempo inicial de validação ou teste.

### 4.4 Modelos Neurais (`ai.models`)
- Modelos herdam de [`BaseTimeSeriesForecaster`](ai/models/base.py) com inicialização ortogonal dos pesos recorrentes e Xavier nas conexões de entrada.
- Camada de normalização [`nn.LayerNorm`](ai/models/gru.py) e dropout sobre o vetor latente final, seguida por cabeça de projeção densa linear com ativação GELU.

### 4.5 Motor de Treinamento (`ai.training.trainer`)
- Implementado em **PyTorch puro**.
- Suporte a perdas robustas como **Huber Loss** ($\text{Smooth } L_1$, resistente a ruídos industriais), MSE e MAE.
- Otimizador AdamW com decaimento de peso (*weight decay*), *gradient clipping* configurável para estabilidade numérica e *Learning Rate Scheduler* (`ReduceLROnPlateau`).
- **Early Stopping** monitorando a perda na validação e restauração automática dos melhores pesos.

### 4.6 Avaliação e Métricas (`ai.evaluation.evaluator`)
- Reverte as predições para as unidades de engenharia originais (ex.: eficiência em porcentagem real).
- Gera previsões detalhadas em formato `.csv`, curvas de perda época a época e painéis comparativos Real vs. Predito com gráficos de dispersão de paridade 1:1.

---

## 5. Instalação e Configuração do Ambiente

Recomenda-se a utilização de um ambiente virtual Python (3.10 a 3.12). O repositório conta com a pasta [`deep-venv`](deep-venv/) previamente estruturada.

```bash
# 1. Ativação do ambiente virtual
source deep-venv/bin/activate

# 2. Instalação das dependências do projeto
pip install -r requirements.txt
```

### Verificação de Aceleração por GPU (CUDA)
Para confirmar que o PyTorch reconhece sua placa de vídeo (como a RTX 3070 Ti):

```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 6. Como Executar

### 6.1 Execução da Análise Exploratória (EDA via Script)

Para gerar e salvar todos os **50 gráficos em PDF individual** da EDA diretamente via terminal (sem necessidade de abrir o Jupyter):

```bash
# Execução padrão (lê data/compressor.csv e salva em results/eda/)
python scripts/run_eda.py

# Opções customizadas (ex: formato PNG ou outro diretório)
python scripts/run_eda.py --data-path data/compressor.csv --output-dir results/eda --format pdf
```

---

### 6.2 Treinamento e Avaliação de Deep Learning (CLI)

O pipeline pode ser executado diretamente pela raiz ou de dentro de `ai/`. O argumento de configuração aceita o padrão posicional, `--config` ou sintaxe direta `--"caminho.yaml"`:

```bash
# Executa a configuração padrão (treina GRU e LSTM e compara resultados)
python run.py --"ai/config.yaml"

# Ou utilizando a flag tradicional --config
python run.py --config ai/config.yaml

# Treinando apenas o modelo GRU
python run.py --"ai/configs/gru.yaml"

# Treinando apenas o modelo LSTM
python run.py --"ai/configs/lstm.yaml"
```

#### Sobrescrita Rápida por Linha de Comando
É possível sobrescrever hiperparâmetros sem alterar o arquivo YAML:

```bash
python run.py --config ai/config.yaml --model gru --epochs 35 --batch-size 64 --lr 0.0005
```

---

### 6.2 Configuração via YAML

O comportamento do pipeline é regido por arquivos declarativos em `.yaml` ([`ai/config.yaml`](ai/config.yaml)):

```yaml
experiment:
  name: "compressor_cbm_forecasting"
  seed: 42
  device: "auto"              # "auto" utiliza GPU se disponível; ou force "cuda" / "cpu"
  output_dir: "results/ai"

data:
  csv_path: "data/compressor.csv"
  reconstruct_poly_eff_dev: true
  drop_all_nan_cols: true

preprocessing:
  target_cols: ["act_poly_eff"] # Alvo principal: Eficiência Politrópica Real (%)
  resample_freq: "5min"         # Fusão na grade regular de 5 minutos
  filter_physical_anomalies: true
  train_ratio: 0.70             # 70% cronológico inicial para treino
  val_ratio: 0.15               # 15% subsequente para validação
  test_ratio: 0.15              # 15% final reservado para teste cego
  split_gap_steps: 1            # Barreira contra contaminação entre partições
  max_ffill_steps: 3            # Limite de imputação causal para frente
  scaler_type: "standard"       # Normalização Z-score com fit exclusivo no treino

windowing:
  lookback: 48                  # 48 passos de 5min = 4 horas de contexto histórico
  horizon: 1                    # Predição de 1 passo à frente (5 min adiante)
  stride_train: 1
  stride_eval: 1

model:
  names: ["gru", "lstm"]        # Executa e compara ambos os modelos
  hidden_size: 64
  num_layers: 2
  dropout: 0.20
  bidirectional: false

training:
  epochs: 25
  batch_size: 128
  lr: 0.001
  weight_decay: 0.0001
  loss: "huber"                 # Huber Loss para imunidade a ruídos industriais
  optimizer: "adamw"
  scheduler: "reduce_on_plateau"
  early_stopping_patience: 6
```

---

## 7. Métricas de Desempenho e Avaliação

O avaliador calcula métricas na **escala física original** (ex.: erro real em percentual de eficiência), permitindo interpretação direta para engenharia de confiabilidade:

$$\begin{aligned}
\text{MAE} &= \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| \\
\text{RMSE} &= \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \\
\text{MAPE (\%)} &= \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \\
\text{sMAPE (\%)} &= \frac{100\%}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2} \\
R^2 &= 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\end{aligned}$$

---

## 8. Resultados e Artefatos Gerados

### 8.1 Artefatos da Análise Exploratória (`results/eda/`)
O notebook [`eda/analise_compressor.ipynb`](eda/analise_compressor.ipynb) e o script [`scripts/run_eda.py`](scripts/run_eda.py) foram configurados para exportar todos os gráficos individualmente em **PDF** para a pasta [`results/eda/`](results/eda/), com subpasta dedicada exclusivamente para cada série temporal:

- **Subpasta Exclusiva de Séries Temporais (`results/eda/series_temporais/`):**
  - Contém o gráfico individual de **cada uma das 12 séries temporais** (`serie_act_press_ratio.pdf`, `serie_exp_press_ratio.pdf`, `serie_act_shaft_power.pdf`, `serie_act_poly_eff.pdf`, etc.) combinando leituras horárias e tendência diária de longo prazo com marcações de paradas.
- **Gráficos de Autocorrelação (ACF) Individuais:**
  - `20_autocorrelacao_acf_<variavel>.pdf`: Gráfico de autocorrelação com defasagem de até 7 dias (168h) e intervalo de confiança de 95% para cada sensor/indicador.
  - `19_dinamica_acf_curto_prazo_30s.pdf`: ACF em alta resolução (0 a 60 min).
  - `21_dinamica_pacf_horaria.pdf`: Autocorrelação Parcial (PACF).
- **Matrizes de Correlação:**
  - `24_matriz_correlacao_pearson.pdf`: Heatmap completo da matriz de correlação linear de Pearson entre todas as 12 variáveis CBM.
  - `24_matriz_correlacao_spearman.pdf`: Heatmap completo da matriz de correlação de postos de Spearman (capturando relações não-lineares monotônicas).
  - `25_agrupamento_hierarquico_dendrograma.pdf`: Dendrograma de agrupamento hierárquico dos sensores baseado em dissimilaridade $1 - |r|$.
- **Demais Gráficos Diagnósticos Individuais:**
  - `01_diagnostico_amostragem_delta_t.pdf` e `02_percentual_valores_ausentes.pdf`: Diagnósticos de amostragem.
  - `04_desvios_cbm_media_movel_24h.pdf` e `05_desvios_cbm_tendencia_7d.pdf`: Evolução dos 4 desvios CBM.
  - `06_zoom_parada_potencia_eixo.pdf` a `08_zoom_parada_glitch_eficiencia.pdf`: Zoom na parada de 24–25/Set.
  - `09_distribuicao_<variavel>.pdf` e `10_boxplots_zscore_padronizados.pdf`: Histogramas, KDE e boxplots.
  - `11_sazonalidade_ciclo_diurno.pdf` a `14_sazonalidade_heatmap_mes_hora.pdf`: Sazonalidades horária, diária e mensal.
  - `15_stl_serie_original.pdf` a `18_stl_residuos.pdf`: Componentes individuais da decomposição STL.
  - `22_dinamica_correlacao_cruzada_lead_lag.pdf` e `23_dinamica_espectro_welch_periodicidades.pdf`: Dinâmica temporal e espectro de Welch.
  - `26_mapa_termodinamico_rp_vs_hp.pdf` a `28_mapa_termodinamico_head_vs_eff.pdf`: Mapas de performance.
  - `29_pca_scree_plot_variancia.pdf` a `31_pca_projecao_latente_tempo.pdf`: Análise de PCA.
  - `32_regimes_operacionais_kmeans_tempo.pdf`: Regimes operacionais K-Means.
  - `33_anomalias_consenso_eficiencia.pdf` a `36_volatilidade_movel_potencia_24h.pdf`: Detecção de anomalias, CUSUM e volatilidade.

---

### 8.2 Artefatos do Pipeline de Deep Learning (`results/ai/`)
A estrutura de saídas dos modelos é estritamente **segmentada por modelo (`gru/` e `lstm/`)**, com gráficos salvos individualmente em formato vetorial **`.pdf`**:

```
results/ai/
├── metrics_comparison.csv             # Tabela comparativa de métricas dos modelos
├── comparison_rmse.pdf                # Gráfico individual comparativo de RMSE
├── comparison_mae.pdf                 # Gráfico individual comparativo de MAE
├── pipeline_report.json               # Relatório estruturado com metadados e auditoria
├── gru/                               # Diretório exclusivo do modelo GRU
│   ├── checkpoints/
│   │   └── best_gru.pt                # Pesos ótimos da rede GRU
│   ├── predictions_val.csv            # Predições e resíduos na validação
│   ├── predictions_test.csv           # Predições e resíduos no teste cego
│   ├── loss_train_val.pdf             # Curva individual de Loss (Treino vs Validação)
│   ├── learning_rate.pdf              # Evolução individual do Learning Rate
│   ├── series_val.pdf                 # Série temporal completa (Validação)
│   ├── zoom_val.pdf                   # Zoom detalhado passo a passo (Validação)
│   ├── parity_val.pdf                 # Gráfico de dispersão e paridade 1:1 (Validação)
│   ├── residuals_val.pdf              # Gráfico de resíduos no tempo (Validação)
│   ├── series_test.pdf                # Série temporal completa (Teste)
│   ├── zoom_test.pdf                  # Zoom detalhado passo a passo (Teste)
│   ├── parity_test.pdf                # Gráfico de paridade 1:1 (Teste)
│   └── residuals_test.pdf             # Gráfico de resíduos no tempo (Teste)
└── lstm/                              # Diretório exclusivo do modelo LSTM
    ├── checkpoints/
    │   └── best_lstm.pt               # Pesos ótimos da rede LSTM
    ├── predictions_val.csv            # Predições e resíduos na validação
    ├── predictions_test.csv           # Predições e resíduos no teste cego
    ├── loss_train_val.pdf             # Curva individual de Loss (Treino vs Validação)
    ├── learning_rate.pdf              # Evolução individual do Learning Rate
    ├── series_val.pdf                 # Série temporal completa (Validação)
    ├── zoom_val.pdf                   # Zoom detalhado passo a passo (Validação)
    ├── parity_val.pdf                 # Gráfico de dispersão e paridade 1:1 (Validação)
    ├── residuals_val.pdf              # Gráfico de resíduos no tempo (Validação)
    ├── series_test.pdf                # Série temporal completa (Teste)
    ├── zoom_test.pdf                  # Zoom detalhado passo a passo (Teste)
    ├── parity_test.pdf                # Gráfico de paridade 1:1 (Teste)
    └── residuals_test.pdf             # Gráfico de resíduos no tempo (Teste)
```
