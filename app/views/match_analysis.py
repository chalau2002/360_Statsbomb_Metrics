from IPython.core.magics import config
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import io
from utils.utils import plot_pass_on_pitch, plot_reception_on_pitch, add_pitch_legend, add_reception_legend, pass_title, reception_title
from data_loader import get_match_events

def show_match_analysis(matches, lbp, rai):
    
    st.title("Análise das Métricas por Jogo")
    st.write("Selecione um jogo para ver as estatísticas e visualizações 360 das métricas desenvolvidas.")
    
    # cria o rótulo identificador do jogo para a caixa de seleção ordenado por data
    matches = matches.copy()
    matches['match_date_dt'] = pd.to_datetime(matches['match_date'])
    matches = matches.sort_values(by='match_date_dt', ascending=True)
    
    matches['match_label'] = matches.apply(
        lambda r: f"{r['home_team']} {r['home_score']} - {r['away_score']} {r['away_team']} ({r['match_date']})", 
        axis=1
    )
    match_options = dict(zip(matches['match_label'], matches['match_id']))
    selected_match_label = st.selectbox("Escolha o Jogo", list(match_options.keys()))
    selected_match_id = match_options[selected_match_label]
    
    match_row = matches[matches['match_id'] == selected_match_id].iloc[0]
    
    # renderiza o placar do jogo utilizando componentes padrão do Streamlit
    with st.container(border=True):
        st.markdown(f"""
        <div style="text-align: center;">
            <h3 style="margin: 0; color: var(--text-color);">{match_row['competition_name']} - Matchweek {match_row['match_week']}</h3>
            <div style="display: flex; justify-content: center; align-items: center; gap: 30px; margin: 15px 0;">
                <div style="font-size: 28px; font-weight: 800; width: 40%; text-align: right; color: var(--text-color);">{match_row['home_team']}</div>
                <div style="font-size: 32px; font-weight: 800; background-color: rgba(128, 128, 128, 0.15); padding: 8px 18px; border-radius: 8px; color: var(--text-color);">{match_row['home_score']} - {match_row['away_score']}</div>
                <div style="font-size: 28px; font-weight: 800; width: 40%; text-align: left; color: var(--text-color);">{match_row['away_team']}</div>
            </div>
            <div style="color: gray; font-size: 14px;">Estádio: {match_row['stadium']} | Data: {match_row['match_date']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # carrega os eventos 360 do jogo diretamente da base de dados
    events_df = get_match_events(selected_match_id)
    
    # filtros na barra lateral (Sidebar) específicos para esta página
    st.sidebar.markdown("### Filtros do Jogo")
    
    half_options = ["Tudo", "1ª Parte", "2ª Parte"]
    selected_half = st.sidebar.selectbox("Fase do Jogo", half_options)
    
    third_options = ["Tudo", "1º Terço", "2º Terço", "3º Terço"]
    selected_third = st.sidebar.selectbox("Zona do Terreno", third_options)

    # filtra os dados com base nos critérios selecionados
    m_lbp = lbp[lbp['match_id'] == selected_match_id].copy()
    m_rai = rai[rai['match_id'] == selected_match_id].copy()
    

    if selected_half == "1ª Parte":
        m_lbp = m_lbp[m_lbp['period'] == 1]
        m_rai = m_rai[m_rai['period'] == 1]
    elif selected_half == "2ª Parte":
        m_lbp = m_lbp[m_lbp['period'] == 2]
        m_rai = m_rai[m_rai['period'] == 2]
        
    if selected_third == "1º Terço":
        m_lbp = m_lbp[m_lbp['pass_x'] < 40]
        m_rai = m_rai[m_rai['rec_x'] < 40]
    elif selected_third == "2º Terço":
        m_lbp = m_lbp[(m_lbp['pass_x'] >= 40) & (m_lbp['pass_x'] < 80)]
        m_rai = m_rai[(m_rai['rec_x'] >= 40) & (m_rai['rec_x'] < 80)]
    elif selected_third == "3º Terço":
        m_lbp = m_lbp[m_lbp['pass_x'] >= 80]
        m_rai = m_rai[m_rai['rec_x'] >= 80]
        
    # dados por equipa para os cards
    home_team = match_row['home_team']
    away_team = match_row['away_team']
    
    m_lbp_unfiltered = lbp[lbp['match_id'] == selected_match_id].copy()
    m_rai_unfiltered = rai[rai['match_id'] == selected_match_id].copy()
    
    if selected_half == "1ª Parte":
        m_lbp_unfiltered = m_lbp_unfiltered[m_lbp_unfiltered['period'] == 1]
        m_rai_unfiltered = m_rai_unfiltered[m_rai_unfiltered['period'] == 1]
    elif selected_half == "2ª Parte":
        m_lbp_unfiltered = m_lbp_unfiltered[m_lbp_unfiltered['period'] == 2]
        m_rai_unfiltered = m_rai_unfiltered[m_rai_unfiltered['period'] == 2]
        
    if selected_third == "1º Terço":
        m_lbp_unfiltered = m_lbp_unfiltered[m_lbp_unfiltered['pass_x'] < 40]
        m_rai_unfiltered = m_rai_unfiltered[m_rai_unfiltered['rec_x'] < 40]
    elif selected_third == "2º Terço":
        m_lbp_unfiltered = m_lbp_unfiltered[(m_lbp_unfiltered['pass_x'] >= 40) & (m_lbp_unfiltered['pass_x'] < 80)]
        m_rai_unfiltered = m_rai_unfiltered[(m_rai_unfiltered['rec_x'] >= 40) & (m_rai_unfiltered['rec_x'] < 80)]
    elif selected_third == "3º Terço":
        m_lbp_unfiltered = m_lbp_unfiltered[m_lbp_unfiltered['pass_x'] >= 80]
        m_rai_unfiltered = m_rai_unfiltered[m_rai_unfiltered['rec_x'] >= 80]
        
    home_lbp_data = home_lbp_data_all = m_lbp_unfiltered[m_lbp_unfiltered['team'] == home_team]
    home_rai_data = home_rai_data_all = m_rai_unfiltered[m_rai_unfiltered['team'] == home_team]
    
    away_lbp_data = away_lbp_data_all = m_lbp_unfiltered[m_lbp_unfiltered['team'] == away_team]
    away_rai_data = away_rai_data_all = m_rai_unfiltered[m_rai_unfiltered['team'] == away_team]

    num_lbp = len(m_lbp)
    num_rai = len(m_rai)
        
    # renderiza os blocos estatísticos lado a lado
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        with st.container(border=True):
            st.subheader(f"{home_team}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Line Breaking Pass Value**")
                n_h_lbp = len(home_lbp_data)
                avg_h_lbp = home_lbp_data['score'].mean() if n_h_lbp > 0 else 0.0
                avg_h_lines = home_lbp_data['lines_broken'].mean() if n_h_lbp > 0 else 0.0
                
                st.metric("Average Score", f"{avg_h_lbp:.2f}")
                st.metric("Passes LB", n_h_lbp)
                st.metric("Linhas Quebradas/Passe LB", f"{avg_h_lines:.2f}")
                
            with c2:
                st.markdown("**Reception Ability Index**")
                n_h_rai = len(home_rai_data)
                avg_h_rai = home_rai_data['score'].mean() if n_h_rai > 0 else 0.0
                avg_h_dens = home_rai_data['difficulty_context'].mean() if n_h_rai > 0 else 0.0
                
                st.metric("Average Score", f"{avg_h_rai:.2f}")
                st.metric("Total Receções", n_h_rai)
                st.metric("Dificuldade Contextual Média", f"{avg_h_dens:.3f}")
        
    with col_t2:
        with st.container(border=True):
            st.subheader(f"{away_team}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Line Breaking Pass Value**")
                n_a_lbp = len(away_lbp_data)
                avg_a_lbp = away_lbp_data['score'].mean() if n_a_lbp > 0 else 0.0
                avg_a_lines = away_lbp_data['lines_broken'].mean() if n_a_lbp > 0 else 0.0
                
                st.metric("Average Score", f"{avg_a_lbp:.2f}")
                st.metric("Passes LB", n_a_lbp)
                st.metric("Linhas Quebradas/Passe LB", f"{avg_a_lines:.2f}")
                
            with c2:
                st.markdown("**Reception Ability Index**")
                n_a_rai = len(away_rai_data)
                avg_a_rai = away_rai_data['score'].mean() if n_a_rai > 0 else 0.0
                avg_a_dens = away_rai_data['difficulty_context'].mean() if n_a_rai > 0 else 0.0
                
                st.metric("Average Score", f"{avg_a_rai:.2f}")
                st.metric("Total Receções", n_a_rai)
                st.metric("Dificuldade Contextual Média", f"{avg_a_dens:.3f}")
    # prepara agrupamentos por jogador para os gráficos interativos de barras
    df_passers = pd.DataFrame(columns=['Jogador', 'Average Score', 'Total Passes LB', 'Média Linhas Quebradas'])
    df_receivers = pd.DataFrame(columns=['Jogador', 'Average Score','Total Receções', 'Média Espaço', 'Média Densidade Defensiva (Op)'])

    if len(m_lbp) > 0:
        df_passers = m_lbp.groupby('player').agg(
            team=('team', 'first'),
            total=('score', 'count'),
            mean_score=('score', 'mean'),
            mean_lines=('lines_broken', 'mean')
        ).reset_index().rename(columns={
            'player': 'Jogador',
            'team': 'Equipa',
            'total': 'Total Passes LB',
            'mean_score': 'Average Score',
            'mean_lines': 'Média Linhas Quebradas'
        })

    if len(m_rai) > 0:
        df_receivers = m_rai.groupby('player').agg(
            team=('team', 'first'),
            total=('score', 'count'),
            mean_score=('score', 'mean'),
            mean_space=('voronoi_area', 'mean'),
            mean_difficulty=('difficulty_context', 'mean')
        ).reset_index().rename(columns={
            'player': 'Jogador',
            'team': 'Equipa',
            'total': 'Total Receções',
            'mean_score': 'Average Score',
            'mean_space': 'Média Espaço',
            'mean_difficulty': 'Dificuldade Contextual Média'
        })

    st.write("### Desempenho Individual de Jogadores")
    col_c1, col_c2 = st.columns(2)

    # altura do contentor = tamanho do LBPV; RAI fica com scroll se tiver mais jogadores
    lbpv_height = max(len(df_passers) * 30, 120)
    rai_height   = max(len(df_receivers) * 30, 120)
    container_height = lbpv_height + 50 

    with col_c1:
        with st.container(border=True):
            st.write("#### LBPV")
            if not df_passers.empty:
                metric_lbp = st.selectbox(
                    "", 
                    ["Average Score", "Total Passes LB", "Média Linhas Quebradas"],
                    key="lbp_select_chart"
                )
                
                # ordena todos os jogadores do maior para o menor
                chart_data = df_passers.sort_values(by=metric_lbp, ascending=False)
                
                # gráfico horizontal do Altair com altura dinâmica baseada no número de jogadores
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X(metric_lbp, type='quantitative', title=metric_lbp,
                            axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
                    y=alt.Y('Jogador', type='nominal', sort='-x', title=None,
                            axis=alt.Axis(labelFontSize=10, labelLimit=300)),
                    color=alt.Color('Equipa', type='nominal', title='Equipa',
                                    scale=alt.Scale(domain=[home_team, away_team], range=['#1f77b4', '#aec7e8'])),
                    tooltip=['Jogador', 'Equipa', metric_lbp]
                ).properties(
                    height=lbpv_height,
                    width='container'
                )
                
                with st.container(height=container_height, border=False):
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum passe registado para os filtros selecionados.")

    with col_c2:
        with st.container(border=True):
            st.write("#### RAI")
            if not df_receivers.empty:
                metric_rai = st.selectbox(
                    "", 
                    ["Average Score", "Total Receções", "Média Espaço", "Dificuldade Contextual Média"],
                    key="rai_select_chart"
                )
                
                # ordena todos os jogadores do maior para o menor
                chart_data = df_receivers.sort_values(by=metric_rai, ascending=False)
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X(metric_rai, type='quantitative', title=metric_rai,
                            axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
                    y=alt.Y('Jogador', type='nominal', sort='-x', title=None,
                            axis=alt.Axis(labelFontSize=10, labelLimit=300)),
                    color=alt.Color('Equipa', type='nominal', title='Equipa',
                                    scale=alt.Scale(domain=[home_team, away_team], range=['#1f77b4', '#aec7e8'])),
                    tooltip=['Jogador', 'Equipa', metric_rai]
                ).properties(
                    height=rai_height,
                    width='container'
                )
                
                # contentor com a mesma altura do LBPV — scroll automático se RAI tiver mais jogadores
                with st.container(height=container_height, border=False):
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Nenhuma receção registada para os filtros selecionados.")

    st.write("### Detalhes e Visualização de Ações Individuais")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        with st.container(border=True):
            if num_lbp > 0:
                m_lbp['select_label'] = m_lbp.apply(
                    lambda r: f"Min {r['minute']} - {r['player']} para {r['pass_recipient']} (Score: {r['score']:.1f})", 
                    axis=1
                )
                lbp_sel = st.selectbox("Escolha um Passe", m_lbp['select_label'].tolist())
            
                selected_rows = m_lbp[m_lbp['select_label'] == lbp_sel]
                if not selected_rows.empty:
                    pass_row = selected_rows.iloc[0]
                else:
                    pass_row = m_lbp.iloc[0]
                
                fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
                title = pass_title(pass_row, matches)
                plot_pass_on_pitch(ax, pass_row, events_df, title=title)
                fig.subplots_adjust(bottom=0.20)
                add_pitch_legend(fig, ncol=3, fontsize=7, anchor=(0.5, 0.02))
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)

                st.image(buf, width=700)
            else:
                st.info("Nenhum passe encontrado para os filtros selecionados.")
        
    with col_plot2:
        with st.container(border=True):
            if num_rai > 0:
                m_rai['select_label'] = m_rai.apply(
                    lambda r: f"Min {r['minute']} - {r['player']} (Score: {r['score']:.2f})", 
                    axis=1
                )
                rai_sel = st.selectbox("Escolha uma Receção", m_rai['select_label'].tolist())
                
                selected_rows = m_rai[m_rai['select_label'] == rai_sel]
                if not selected_rows.empty:
                    rec_row = selected_rows.iloc[0]
                else:
                    rec_row = m_rai.iloc[0]
                
                fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
                title = reception_title(rec_row, matches)
                plot_reception_on_pitch(ax, rec_row, events_df, title=title)
                fig.subplots_adjust(bottom=0.20)
                add_reception_legend(fig, ncol=3, fontsize=7, anchor=(0.5, 0.02))
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)

                st.image(buf, width=700)
            else:
                st.info("Nenhuma receção sob pressão encontrada para os filtros selecionados.")
