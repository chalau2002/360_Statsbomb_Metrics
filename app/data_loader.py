import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import getpass

# cria o motor de ligação SQLAlchemy para a base de dados
@st.cache_resource
def get_db_engine():
    import os
    db_url = None
    
    # tenta ler do Streamlit Secrets
    try:
        if "DATABASE_URL" in st.secrets:
            db_url = st.secrets["DATABASE_URL"]
    except Exception:
        pass
        
    # tenta ler das variáveis de ambiente
    if not db_url:
        db_url = os.environ.get("DATABASE_URL")
        
    # fallback para a base de dados local
    if not db_url:
        db_url = "postgresql://localhost:5432/statsbomb_db"
        
    # o SQLAlchemy 2.0 requer 'postgresql://' em vez de 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

# carrega as tabelas principais da base de dados
@st.cache_data
def load_base_tables():
    engine = get_db_engine()
    competitions = pd.read_sql("SELECT * FROM competitions", engine)
    matches = pd.read_sql("SELECT * FROM matches", engine)
    lineups = pd.read_sql("SELECT * FROM lineups", engine)
    lbp = pd.read_sql("SELECT * FROM line_breaking_passes", engine)
    rai = pd.read_sql("SELECT * FROM reception_ability_index", engine)
    return competitions, matches, lineups, lbp, rai

# carrega os eventos de um jogo específico
@st.cache_data
def get_match_events(match_id):
    engine = get_db_engine()
    query = """
        SELECT id, match_id, teammate, actor, keeper, location_y 
        FROM events 
        WHERE match_id = %s
    """
    return pd.read_sql(query, engine, params=(int(match_id),))


# determina as posições mais frequentes dos jogadores
@st.cache_data
def get_player_positions():
    engine = get_db_engine()
    query = """
        SELECT player, position, count(*) as count
        FROM events
        WHERE player IS NOT NULL 
          AND position IS NOT NULL
        GROUP BY player, position
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return {}
        
    # obtém o índice do valor máximo de contagem para cada jogador
    idx = df.groupby('player')['count'].idxmax()
    df_modes = df.loc[idx]
    
    return dict(zip(df_modes['player'], df_modes['position']))
