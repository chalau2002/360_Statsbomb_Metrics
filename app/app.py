import sys
import os
import streamlit as st
import pandas as pd

# garante que a pasta pai está no sys.path para importar utilitários
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# importações das funções para carregar dados e páginas
from data_loader import load_base_tables, get_player_positions
from views.match_analysis import show_match_analysis
from views.competition_analysis import show_competition_analysis
from views.player_comparison import show_player_comparison
from views.scatterplots import show_scatterplots

# configuração da página padrão do Streamlit
st.set_page_config(
    page_title="Dashboard de Métricas 360",
    layout="wide",
    initial_sidebar_state="expanded"
)

# título da aplicação
st.sidebar.title("Dashboard de Métricas 360")

# menu de navegação
page = st.sidebar.selectbox(
    "Navegação",
    ["Análise de Jogo", "Análise de Competição", "Comparação de Jogadores", "Dispersão e Perfis"]
)

# carrega todas as tabelas base e configurações dos jogadores
with st.spinner("A carregar dados do PostgreSQL..."):
    competitions, matches, lineups, lbp, rai = load_base_tables()
    player_positions = get_player_positions()

# pré-calcula o DataFrame consolidado de jogadores para as diferentes páginas
p_minutes = lineups.groupby('player_name')['minutes_played'].sum().reset_index(name='minutes').rename(columns={'player_name': 'player'})
p_matches = lineups.groupby('player_name')['match_id'].nunique().reset_index(name='matches').rename(columns={'player_name': 'player'})

# agrega os dados por jogador para a métrica LBPV
lbp_agg = lbp.groupby('player').agg(
    lbp_count=('score', 'count'), 
    lbp_avg_score=('score', 'mean'), 
    avg_lines=('lines_broken', 'mean')
).reset_index()

# agrega os dados por jogador para a métrica RAI
rai_agg = rai.groupby('player').agg(
    rai_count=('score', 'count'), 
    rai_avg_score=('score', 'mean'), 
    avg_space=('voronoi_area', 'mean'),
    avg_difficulty=('difficulty_context', 'mean')
).reset_index()

# une todos os DataFrames agregados e cria o DataFrame consolidado de jogadores
player_df = p_minutes
player_df = player_df.merge(p_matches, on='player', how='outer')
player_df = player_df.merge(lbp_agg, on='player', how='outer')
player_df = player_df.merge(rai_agg, on='player', how='outer')

# atribui a equipa mais frequente ao jogador
player_team = lineups.groupby('player_name')['team_name'].agg(lambda x: x.value_counts().index[0]).reset_index()
player_team = player_team.rename(columns={'player_name': 'player', 'team_name': 'team'})
player_df = player_df.merge(player_team, on='player', how='left')

player_df['position'] = player_df['player'].map(player_positions).fillna('Desconhecido')
player_df['minutes'] = player_df['minutes'].fillna(0).astype(float)
player_df['lbp_count'] = player_df['lbp_count'].fillna(0).astype(int)
player_df['rai_count'] = player_df['rai_count'].fillna(0).astype(int)


# navegação entre páginas com os dataframes necessários
if page == "Análise de Jogo":
    show_match_analysis(matches, lbp, rai)

elif page == "Análise de Competição":
    show_competition_analysis(matches, lineups, lbp, rai, player_positions)

elif page == "Comparação de Jogadores":
    show_player_comparison(lineups, lbp, rai, player_positions)

elif page == "Dispersão e Perfis":
    show_scatterplots(player_df, rai, player_positions)
