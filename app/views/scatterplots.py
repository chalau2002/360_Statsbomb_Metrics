import streamlit as st
import plotly.express as px
import pandas as pd
from utils.utils import POS_TO_GROUP

def show_scatterplots(player_df, rai, player_positions):
    st.title("Dispersão e Perfis de Jogadores")

    # mapeamento de posição
    df = player_df.copy()
    raw_pos = df['player'].map(player_positions).fillna('Desconhecido')
    df['position_group'] = raw_pos.map(POS_TO_GROUP).fillna('Outros/Substitutos')

    # remove guarda-redes e substitutos
    df = df[~df['position_group'].isin(['Guarda-redes', 'Outros/Substitutos'])]

    # filtros na barra lateral
    st.sidebar.markdown("### Filtros do Gráfico")
    min_min = st.sidebar.slider("Mínimo Minutos Jogados", 0, int(df['minutes'].max()), 90)
    min_lbp = st.sidebar.slider("Mínimo Passes LB", 0, int(df['lbp_count'].max()), 4)
    min_rai = st.sidebar.slider("Mínimo Receções", 0, int(df['rai_count'].max()), 20)

    unique_groups = sorted(df['position_group'].unique())
    selected_pos = st.sidebar.multiselect("Posições", unique_groups, default=unique_groups)

    # aplica filtros
    df_sc = df[
        (df['minutes'] >= min_min) &
        (df['lbp_count'] >= min_lbp) &
        (df['rai_count'] >= min_rai) &
        (df['position_group'].isin(selected_pos))
    ].copy()

    tab1, tab2 = st.tabs(["Dispersão", "Matriz de Perfis de Receção"])

    # gráfico de dispersão
    with tab1:
        st.write("### Gráfico de Dispersão")

        metrics_dict = {
            "Passes Line-Breaking Totais": "lbp_count",
            "LBPV Average Score": "lbp_avg_score",
            "Linhas Quebradas / Passe LB": "avg_lines",
            "Receções Totais": "rai_count",
            "RAI Average Score": "rai_avg_score",
            "Controlo de Espaço (m²)": "avg_space",
            "Dificuldade Contextual": "avg_difficulty",
        }

        metric_keys = list(metrics_dict.keys())
        x_label = st.selectbox("Selecione Eixo X", metric_keys, index=metric_keys.index("LBPV Average Score"))
        y_label = st.selectbox("Selecione Eixo Y", metric_keys, index=metric_keys.index("RAI Average Score"))

        x_col = metrics_dict[x_label]
        y_col = metrics_dict[y_label]

        plot_df = df_sc.dropna(subset=[x_col, y_col])

        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                hover_name="player",
                hover_data={
                    "team": True,
                    "minutes": ":.0f",
                    "position_group": True,
                    x_col: ":.2f",
                    y_col: ":.2f",
                },
                labels={x_col: x_label, y_col: y_label, "position_group": "Posição"},
                title=f"{x_label} vs {y_label}",
                opacity=0.85,
            )
            fig.update_traces(marker=dict(size=10, color="#1f77b4"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum jogador encontrado com os filtros selecionados.")

    # matriz de perfis de receção
    with tab2:
        st.write("### Matriz de Perfis de Receção (RAI)")
        st.write("Análise baseada na área Voronoi útil e na dificuldade do contexto defensivo.")

        # medianas calculadas sobre os eventos raw (centro fixo da matriz)
        v_median = rai['voronoi_area_norm'].median()
        d_median = rai['difficulty_context'].median()

        def classify_profile(row):
            v, d = row['voronoi_area_norm'], row['difficulty_context']
            if v >= v_median and d >= d_median:
                return 'Advanced Space Receiver'
            elif v < v_median and d >= d_median:
                return 'Pressured Target'
            elif v >= v_median and d < d_median:
                return 'Free-Space Receiver'
            return 'Neutral / Peripheral'

        # agrega para médias por jogador
        rai_player = rai.groupby('player').agg(
            difficulty_context=('difficulty_context', 'mean'),
            voronoi_area_norm=('voronoi_area_norm', 'mean'),
            receptions=('score', 'count'),
        ).reset_index()

        rai_player = rai_player.merge(
            df[['player', 'team', 'position_group', 'minutes', 'lbp_count']],
            on='player', how='inner'
        )

        df_matrix = rai_player[
            (rai_player['minutes'] >= min_min) &
            (rai_player['lbp_count'] >= min_lbp) &
            (rai_player['receptions'] >= min_rai) &
            (rai_player['position_group'].isin(selected_pos))
        ].copy()

        if not df_matrix.empty:
            df_matrix['profile'] = df_matrix.apply(classify_profile, axis=1)


            color_map = {
                'Advanced Space Receiver': '#1f77b4',
                'Pressured Target': '#d62728',
                'Free-Space Receiver': '#2ca02c',
                'Neutral / Peripheral': '#7f7f7f',
            }

            fig2 = px.scatter(
                df_matrix,
                x='difficulty_context',
                y='voronoi_area_norm',
                color='profile',
                color_discrete_map=color_map,
                hover_name='player',
                hover_data={
                    'team': True,
                    'position_group': True,
                    'receptions': True,
                    'difficulty_context': ':.3f',
                    'voronoi_area_norm': ':.3f',
                    'profile': False,
                },
                labels={
                    'difficulty_context': 'Contexto de Dificuldade (Média)',
                    'voronoi_area_norm': 'Área Voronoi Normalizada (Média)',
                    'position_group': 'Posição',
                },
                title="Matriz de Perfis de Receção (RAI)",
                opacity=0.9,
            )
            fig2.update_traces(marker=dict(size=11))
            fig2.add_vline(x=d_median, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.add_hline(y=v_median, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(legend_title_text="Perfil")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
            **Quadrantes da Matriz:**
            * **Advanced Space Receiver** (Superior Direito): Recebe em zonas de alta dificuldade contextual mas consegue gerar espaço (Voronoi).
            * **Pressured Target** (Inferior Direito): Recebe sob alta pressão e em espaço curtíssimo (pivôs / avançados de referência).
            * **Free-Space Receiver** (Superior Esquerdo): Recebe com espaço em contextos de menor pressão.
            * **Neutral / Peripheral** (Inferior Esquerdo): Receções de menor relevância ou periféricas.
            """)
        else:
            st.info("Nenhum jogador encontrado com os filtros selecionados.")
