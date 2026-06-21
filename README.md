# 360 StatsBomb Metrics

Este projeto é uma ferramenta de suporte analítico com foco na análise de métricas táticas avançadas geradas a partir de dados StatsBomb 360 e eventos. O objetivo principal é quantificar o desempenho de jogadores e equipas através de duas métricas originais:

- **LBPV (Line-Breaking Pass Value)**: Avalia o impacto e a qualidade dos passes que quebram linhas defensivas adversárias, considerando a progressão espacial, defesas ultrapassados e outcome da jogada.
- **RAI (Reception Ability Index)**: Mede a capacidade de receção de bola sob pressão dentro do bloco adversário, ponderando a área de Voronoi controlada pelo recetor no momento da receção e calculando um índice de dificuldade associada à mesma.

A aplicação é constituída por uma pipeline de processamento de dados (Notebooks Jupyter) e um dashboard interativo (Streamlit) que consome os dados a partir de uma base de dados PostgreSQL (local ou remota).

---

## Estrutura do Projeto

```
360_Statsbomb_Metrics/
│
├── data/
│   ├── raw/                        # Dados originais em Parquet (gerados pelo notebook 01)
│   │   ├── competitions.parquet
│   │   ├── matches.parquet
│   │   ├── lineups.parquet
│   │   └── events.parquet
│   └── processed/                  # Dados processados e tabelas finais (gerados pelo notebook 02)
│
├── notebooks/                      # Notebooks Jupyter — executar por ordem
│   ├── 01_fetch_statsbomb_data.ipynb     # Descarrega dados da API StatsBomb
│   ├── 02_process_data.ipynb             # Processa os dados e cria as métricas LBPV e RAI
│   ├── 03_LBPV_analysis.ipynb            # Análise exploratória da métrica LBPV
│   ├── 04_RAI_analysis.ipynb             # Análise exploratória da métrica RAI
│   └── 05_create_postgres_database.ipynb # Cria e popula a base de dados PostgreSQL
│
├── app/                            # Aplicação Streamlit
│   ├── app.py                      # Ponto de entrada da app
│   ├── data_loader.py              # Ligação à BD e carregamento de dados
│   └── views/                      # Páginas da aplicação
│       ├── match_analysis.py       # Análise das Métricas por Jogo
│       ├── competition_analysis.py # Análise das Métricas por Competição
│       ├── player_comparison.py    # Comparação de Jogadores
│       └── scatterplots.py         # Gráfico Dispersão & Perfis
│
├── utils/
│   └── utils.py                    # Funções auxiliares partilhadas
│
├── .streamlit/
│   └── secrets.toml                # Credenciais da base de dados (não versionado)
│
├── requirements.txt
└── README.md
```

---

## Configuração do Ambiente

### 1. Criar e ativar o ambiente virtual

```bash
# Na raiz do projeto
python -m venv .venv

# Ativar (macOS/Linux)
source .venv/bin/activate

# Ativar (Windows)
.venv\Scripts\activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Registar o kernel do Jupyter (para correr os notebooks com o .venv)

```bash
python -m ipykernel install --user --name=360metrics --display-name "Python (360metrics)"
```

---

## Pipeline de Dados — Executar os Notebooks por Ordem

### Notebook 01 — Fetch StatsBomb Data
```
notebooks/01_fetch_statsbomb_data.ipynb
```
Descarrega os dados brutos da API pública StatsBomb (`statsbombpy`) e guarda-os em `data/raw/` no formato Parquet.

### Notebook 02 — Process Data
```
notebooks/02_process_data.ipynb
```
Processa os dados de todas as tabelas, calcula as métricas **LBPV** e **RAI** e guarda os resultados em `data/processed/`.

### Notebook 03 — LBPV Analysis
```
notebooks/03_LBPV_analysis.ipynb
```
Análise exploratória da métrica **Line-Breaking Pass Value (LBPV)**: distribuições, rankings e visualizações no campo.

### Notebook 04 — RAI Analysis
```
notebooks/04_RAI_analysis.ipynb
```
Análise exploratória da métrica **Reception Ability Index (RAI)**: distribuições, perfis de receção e comparação entre jogadores.

### Notebook 05 — Create PostgreSQL Database
```
notebooks/05_create_postgres_database.ipynb
```
Cria as tabelas na base de dados PostgreSQL e popula-as com os dados processados. **A app Streamlit lê exclusivamente desta base de dados.**

O notebook suporta dois modos de execução (configurável no topo do ficheiro):

| Modo | Variável | Descrição |
|---|---|---|
| **Local** | `LOCAL = True` | Liga a `postgresql://localhost:5432/statsbomb_db` |
| **Remoto** | `LOCAL = False` | Usa a `connection string` definida nos `secrets.toml` |

> **Modo Local**: Certifique-se de que o PostgreSQL está ativo antes de executar:
> ```bash
> brew services start postgresql   # macOS
> ```

## Correr a Aplicação Streamlit

Com o ambiente virtual ativo e a base de dados configurada:

```bash
# A partir da pasta app/
cd app
streamlit run app.py
```

## Acesso à Aplicação

- Ambiente local: `http://localhost:8501`
- Deploy em produção: https://360-statsbomb-metrics.streamlit.app/
