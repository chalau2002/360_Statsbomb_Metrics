import streamlit as st
import pandas as pd
from utils.utils import POS_TO_GROUP

def show_competition_analysis(matches, lineups, lbp, rai, player_positions):
   
    st.title("Estatísticas Gerais da Competição")
    st.write("Analise o desempenho geral dos jogadores e equipas na competição. Para ordenar as colunas basta clicar nos cabeçalhos.")
    
    # agrega estatísticas ao nível do jogador
    p_minutes = lineups.groupby('player_name')['minutes_played'].sum().reset_index(name='minutes')
    p_matches = lineups.groupby('player_name')['match_id'].nunique().reset_index(name='matches')
    
    lbp_agg = lbp.groupby('player').agg(
        lbp_count=('score', 'count'),
        lbp_avg_score=('score', 'mean'),
        avg_lines=('lines_broken', 'mean')
    ).reset_index()
    
    rai_agg = rai.groupby('player').agg(
        rai_count=('score', 'count'),
        rai_avg_score=('score', 'mean'),
        avg_space=('voronoi_area', 'mean'),
        avg_difficulty=('difficulty_context', 'mean')
    ).reset_index()
    
    # une os conjuntos de dados dos jogadores
    p_df = p_minutes.merge(p_matches, on='player_name', how='outer')
    p_df = p_df.rename(columns={'player_name': 'player'})
    p_df = p_df.merge(lbp_agg, on='player', how='outer')
    p_df = p_df.merge(rai_agg, on='player', how='outer')
    
    # mapeamento de equipas
    player_team = lineups.groupby('player_name')['team_name'].agg(lambda x: x.value_counts().index[0]).reset_index()
    player_team = player_team.rename(columns={'player_name': 'player', 'team_name': 'team'})
    p_df = p_df.merge(player_team, on='player', how='left')
    
    # preenche os valores nulos com 0
    p_df['lbp_count'] = p_df['lbp_count'].fillna(0).astype(int)
    p_df['rai_count'] = p_df['rai_count'].fillna(0).astype(int)
    p_df['minutes'] = p_df['minutes'].fillna(0).astype(float)
    p_df['matches'] = p_df['matches'].fillna(0).astype(int)
    
    # mapeia a posição usando o mapeamento centralizado
    raw_pos = p_df['player'].map(player_positions).fillna('Desconhecido')
    p_df['position'] = raw_pos.map(POS_TO_GROUP).fillna('Outros/Substitutos')

    # remove guarda-redes e outros/substitutos
    p_df = p_df[~p_df['position'].isin(['Guarda-redes', 'Outros/Substitutos'])]
    
    # filtros na barra lateral
    st.sidebar.markdown("### Filtros da Competição")
    min_min = st.sidebar.slider("Mínimo Minutos Jogados", 0, int(p_df['minutes'].max()), 90)
    min_lbp = st.sidebar.slider("Mínimo Passes LB", 0, int(p_df['lbp_count'].max()), 0)
    min_rai = st.sidebar.slider("Mínimo Receções", 0, int(p_df['rai_count'].max()), 0)
    
    unique_positions = sorted(p_df['position'].unique())
    selected_pos = st.sidebar.multiselect("Posições", unique_positions, default=unique_positions)
    
    # aplica os filtros ao conjunto de dados de jogadores
    filt_players = p_df[
        (p_df['minutes'] >= min_min) &
        (p_df['lbp_count'] >= min_lbp) &
        (p_df['rai_count'] >= min_rai) &
        (p_df['position'].isin(selected_pos))
    ]
        
    st.write("### Estatísticas de Jogadores")
    st.dataframe(
        filt_players[[
            'player', 'team', 'position', 'matches', 'minutes', 
            'lbp_count', 'lbp_avg_score', 'avg_lines', 
            'rai_count', 'rai_avg_score', 'avg_space', 'avg_difficulty'
        ]].sort_values(by='lbp_count', ascending=False).rename(columns={
            'player': 'Jogador',
            'team': 'Equipa',
            'position': 'Posição',
            'matches': 'Jogos',
            'minutes': 'Minutos',
            'lbp_count': 'Passes LB',
            'avg_lines': 'Linhas Quebradas/Passe LB',
            'lbp_avg_score': 'LBPV Médio',
            'rai_count': 'Receções',
            'avg_space': 'Espaço Médio (m²)',
            'avg_difficulty': 'Dificuldade Contextual',
            'rai_avg_score': 'RAI Médio'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.write("### Estatísticas de Equipas")
    
    # contagem de jogos por equipa
    team_matches_counts = pd.concat([matches['home_team'], matches['away_team']]).value_counts().reset_index()
    team_matches_counts.columns = ['team', 'matches']
    
    team_lbp = lbp.groupby('team').agg(
        t_lbp_count=('score', 'count'),
        t_lbp_avg=('score', 'mean')
    ).reset_index()
    
    team_rai = rai.groupby('team').agg(
        t_rai_count=('score', 'count'),
        t_rai_avg=('score', 'mean')
    ).reset_index()
    
    team_df = team_matches_counts.merge(team_lbp, on='team', how='outer')
    team_df = team_df.merge(team_rai, on='team', how='outer')
    team_df = team_df.fillna(0)
    
    st.dataframe(
        team_df.rename(columns={
            'team': 'Equipa',
            'matches': 'Jogos',
            't_lbp_count': 'Passes LB Totais',
            't_lbp_avg': 'LBPV Médio',
            't_rai_count': 'Receções Totais',
            't_rai_avg': 'RAI Médio'
        }),
        use_container_width=True,
        hide_index=True
    )
