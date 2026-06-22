import streamlit as st
import streamlit.components.v1 as components


def show_player_comparison(lineups, lbp, rai, player_positions):
    
    st.title("Comparação Detalhada de Jogadores")
    st.write("Selecione dois jogadores para comparar o seu rendimento nas métricas desenvolvidas.")

    # recalcula a tabela de base de jogadores
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

    p_df = p_minutes.merge(p_matches, on='player_name', how='outer').rename(columns={'player_name': 'player'})
    p_df = p_df.merge(lbp_agg, on='player', how='outer')
    p_df = p_df.merge(rai_agg, on='player', how='outer')

    player_team = lineups.groupby('player_name')['team_name'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index().rename(columns={'player_name': 'player', 'team_name': 'team'})
    p_df = p_df.merge(player_team, on='player', how='left')
    p_df['position'] = p_df['player'].map(player_positions).fillna('Desconhecido')

    p_df['lbp_count'] = p_df['lbp_count'].fillna(0).astype(int)
    p_df['rai_count'] = p_df['rai_count'].fillna(0).astype(int)

    all_players = sorted(p_df['player'].dropna().unique())

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        player_a = st.selectbox("Selecionar Jogador A", all_players, index=0)
    with col_p2:
        player_b = st.selectbox("Selecionar Jogador B", all_players, index=min(1, len(all_players) - 1))

    p_a = p_df[p_df['player'] == player_a].iloc[0]
    p_b = p_df[p_df['player'] == player_b].iloc[0]

    def safe(row, col, default=0.0):
        try:
            v = row[col]
            return float(v) if v == v else default
        except Exception:
            return default

    # valores das métricas
    metrics = {
        'lbp_count':     (int(p_a['lbp_count']), int(p_b['lbp_count']), True,  "Passes Line-Breaking Totais", "{}"),
        'lbp_avg_score': (safe(p_a, 'lbp_avg_score'), safe(p_b, 'lbp_avg_score'), True,  "LBPV Average Score", "{:.2f}"),
        'avg_lines':     (safe(p_a, 'avg_lines'), safe(p_b, 'avg_lines'), True,  "Linhas Quebradas / Passe LB", "{:.2f}"),
        'rai_count':     (int(p_a['rai_count']), int(p_b['rai_count']), True,  "Receções sob Pressão", "{}"),
        'rai_avg_score': (safe(p_a, 'rai_avg_score'), safe(p_b, 'rai_avg_score'), True,  "RAI Average Score", "{:.2f}"),
        'avg_space':     (safe(p_a, 'avg_space'), safe(p_b, 'avg_space'), True,  "Controlo de Espaço (m²)", "{:.1f}"),
        'avg_difficulty': (safe(p_a, 'avg_difficulty'), safe(p_b, 'avg_difficulty'), True,  "Dificuldade Contextual", "{:.3f}"),
    }

    GREEN = "rgba(0,180,80,0.12)"
    RED   = "rgba(220,50,50,0.10)"
    TIE   = "rgba(128,128,128,0.07)"

    def row_color(val_a, val_b, higher_is_better):
        if val_a == val_b:
            return TIE, TIE
        if higher_is_better:
            return (GREEN, RED) if val_a > val_b else (RED, GREEN)
        else:
            return (GREEN, RED) if val_a < val_b else (RED, GREEN)

    # separa métricas por secção
    lbp_keys = ['lbp_count', 'lbp_avg_score', 'avg_lines']
    rai_keys = ['rai_count', 'rai_avg_score', 'avg_space', 'avg_difficulty']

    def metric_html(label, val_a, val_b, fmt, higher_is_better):
        col_a, col_b = row_color(val_a, val_b, higher_is_better)
        try:
            str_a = fmt.format(val_a)
            str_b = fmt.format(val_b)
        except Exception:
            str_a = str(val_a)
            str_b = str(val_b)
        return f"""
        <tr>
            <td style="background:{col_a}; text-align:center; padding:8px 10px; border-radius:6px; font-weight:600;">{str_a}</td>
            <td style="text-align:center; padding:8px 6px; color:rgba(128,128,128,0.8); font-size:12px;">{label}</td>
            <td style="background:{col_b}; text-align:center; padding:8px 10px; border-radius:6px; font-weight:600;">{str_b}</td>
        </tr>"""

    # html da tabela
    rows_lbp = "".join(
        metric_html(metrics[k][3], metrics[k][0], metrics[k][1], metrics[k][4], metrics[k][2])
        for k in lbp_keys
    )
    rows_rai = "".join(
        metric_html(metrics[k][3], metrics[k][0], metrics[k][1], metrics[k][4], metrics[k][2])
        for k in rai_keys
    )

    html_content = f"""
    <style>
      body {{ margin: 0; font-family: sans-serif; }}
      .cmp-table {{ width:100%; border-collapse:separate; border-spacing:0 5px; table-layout:fixed; }}
      .cmp-table td, .cmp-table th {{ vertical-align:middle; overflow:hidden; }}
      .cmp-header {{ text-align:center; font-size:14px; font-weight:700; padding:4px 0 2px 0; }}
      .section-label {{ text-align:center; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#888; padding: 14px 0 4px 0; }}
    </style>
    <table class="cmp-table">
      <colgroup>
        <col style="width:38%"/>
        <col style="width:24%"/>
        <col style="width:38%"/>
      </colgroup>
      <thead>
        <tr>
          <th class="cmp-header">{player_a}</th>
          <th></th>
          <th class="cmp-header">{player_b}</th>
        </tr>
        <tr>
          <td style="text-align:center; font-size:12px; color:#888; padding-bottom:8px;">{p_a['team']} &middot; {p_a['position']}<br/>{int(p_a['matches'])} jogos &middot; {int(p_a['minutes'])} min</td>
          <td></td>
          <td style="text-align:center; font-size:12px; color:#888; padding-bottom:8px;">{p_b['team']} &middot; {p_b['position']}<br/>{int(p_b['matches'])} jogos &middot; {int(p_b['minutes'])} min</td>
        </tr>
      </thead>
      <tbody>
        <tr><td colspan="3" class="section-label">Métricas de Passe — LBPV</td></tr>
        {rows_lbp}
        <tr><td colspan="3" class="section-label">Métricas de Receção — RAI</td></tr>
        {rows_rai}
      </tbody>
    </table>
    """

    components.html(html_content, height=520, scrolling=False)
