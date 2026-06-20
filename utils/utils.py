import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mplsoccer import Pitch

# Constants for defensive-line clustering and line breaking
LINE_GAP_THRESHOLD = 4.5    # gap em X > 4.5m → nova linha defensiva
MIN_LINE_SIZE      = 2      # uma linha necessita de pelo menos 2 defesas
MIN_START_DIST     = 5.0    # o passador deve estar a pelo menos 5m do ponto de interseção


# Palete de cores

COL_BROKEN  = '#facc15'  # amarelo — linha quebrada / respetivos defesas
COL_IN_PATH = '#f97316'  # laranja — linha no trajeto mas não quebrada
COL_OUT     = '#64748b'  # cinzento — linha fora do corredor do passe
COL_PASSER  = '#22c55e'  # verde — passador
COL_PASS    = '#22d3ee'  # azul claro — trajetória do passe



# Pitch
def draw_pitch(ax, bg='white', line_color='#374151'):
    Pitch(
        pitch_type='statsbomb',
        pitch_color=bg,
        line_color=line_color,
        linewidth=1.5,
        line_alpha=0.85,
        goal_type='box',
    ).draw(ax=ax)


# Helpers para legend
def pitch_legend_handles():
    return [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_PASSER,
                      markeredgecolor='white', markersize=9, label='Passer'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_BROKEN,
                      markersize=8, label='Defender — broken line'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_IN_PATH,
                      markersize=8, label='Defender — in-path line'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_OUT,
                      markersize=7, label='Defender — out of path'),
        mlines.Line2D([0], [0], color=COL_PASS, linewidth=2.5,
                      label='Pass trajectory'),
        mlines.Line2D([0], [0], color=COL_BROKEN, linewidth=2.0, linestyle='--',
                      label='Broken line'),
        mlines.Line2D([0], [0], color=COL_IN_PATH, linewidth=1.6,
                      linestyle=(0, (4, 2)), label='Unbroken line (in path)'),
        mlines.Line2D([0], [0], color=COL_OUT, linewidth=1.0, linestyle=':',
                      label='Line out of path'),
        mlines.Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_BROKEN,
                      markersize=6, label='Line crossing point'),
    ]

# Adicionar legenda do pitch na parte inferior central da figura.
def add_pitch_legend(fig, ncol=5, fontsize=9, anchor=(0.5, 0.0)):
    fig.legend(
        handles=pitch_legend_handles(),
        loc='lower center',
        ncol=ncol,
        fontsize=fontsize,
        framealpha=0.0,
        edgecolor='none',
        bbox_to_anchor=anchor,
    )


# Title helper
def pass_title(row, matches_df=None):
    player  = row.get('player', 'Unknown')
    team    = row.get('team', '')
    minute  = int(row.get('minute', 0))
    lines   = int(row['lines_broken'])
    defs    = int(row['n_defenders_bypassed'])
    dist    = float(row.get('distance_advanced', row['end_x'] - row['pass_x']))

    ov      = row['outcome_value']
    outcome = 'Goal' if ov == 1.0 else (f'xG={ov:.2f}' if ov > 0 else 'No shot')

    match_str = ''
    if matches_df is not None:
        m = matches_df[matches_df['match_id'] == row['match_id']]
        if not m.empty:
            r = m.iloc[0]
            match_str = (
                f"\n{r['home_team']} {int(r['home_score'])}"
                f"–{int(r['away_score'])} {r['away_team']}"
            )

    return (
        f"{player} ({team[:3].upper()})  Score: {row['score']:.2f}/10\n"
        f"Min {minute}  |  {lines} line(s)  |  {defs} def bypassed"
        f"  |  {dist:.1f}m  |  {outcome}"
        f"{match_str}"
    )


# funçao para extrair coordenadas (x, y) de um array/lista
def _extract_xy(arr):
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.nan, np.nan
    arr = np.asarray(arr, dtype=float)
    return float(arr[0]), float(arr[1])


# calcular valor posicional da origem do passe com base na coordenada X
def _zone_value(pass_x: float) -> float:
    # aplicar função sigmoide para atribuir maior valor a passes realizados em zonas mais avançadas do terreno, mantendo o output entre 0 e 1
    return 1.0 / (1.0 + np.exp(-0.08 * (pass_x - 70.0)))


# converter timestamp para segundos totais
def _timestamp_to_seconds(ts):
    try:
        parts = str(ts).split(":")
        # converter cada componente para formato numérico
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        # calcular tempo total em segundos
        return h * 3600 + m * 60 + s
    # devolver NaN caso o timestamp seja inválido
    except Exception:
        return np.nan


# agrupar adversários em clusters de linha defensiva com base na proximidade no eixo X (sem limite mínimo de jogadores)
def _cluster_defenders_by_x(defenders_xy, gap_threshold=LINE_GAP_THRESHOLD):
    if not defenders_xy:
        return []
    sd = sorted(defenders_xy, key=lambda p: p[0])
    clusters, current = [], [sd[0]]
    for prev, curr in zip(sd, sd[1:]):
        if curr[0] - prev[0] > gap_threshold:
            clusters.append(current)
            current = []
        current.append(curr)
    clusters.append(current)
    return clusters


# agrupar adversários em linhas defensivas com base na proximidade no eixo X (mantendo apenas linhas com número mínimo de jogadores)
def _cluster_defenders_into_lines(defenders_xy: list) -> list:
    if not defenders_xy:
        return []
    clusters = _cluster_defenders_by_x(defenders_xy, LINE_GAP_THRESHOLD)
    return [ln for ln in clusters if len(ln) >= MIN_LINE_SIZE]


# calcular a coordenada Y da trajetória do passe num determinado ponto X
def _pass_y_at_x(pass_x, pass_y, end_x, end_y, target_x):
    # evitar divisão por zero em passes verticais
    if abs(end_x - pass_x) < 1e-6:
        return pass_y
    # calcular posição relativa do ponto target_x na trajetória do passe
    t = (target_x - pass_x) / (end_x - pass_x)
    # interpolar coordenada Y correspondente ao ponto target_x
    return pass_y + t * (end_y - pass_y)


# verificar se a trajetória do passe atravessa o espaço entre dois adversários de uma linha
def _line_is_broken(pass_x, pass_y, end_x, end_y, cluster):
    """True if the pass cuts through a gap between defenders in *cluster*."""
    line_x = np.mean([p[0] for p in cluster])
    if not (pass_x <= line_x <= end_x):
        return False
    
    y_at_line = _pass_y_at_x(pass_x, pass_y, end_x, end_y, line_x)
    dist_to_intersection = np.sqrt(
        (line_x - pass_x) ** 2 + (y_at_line - pass_y) ** 2
    )
    if dist_to_intersection < MIN_START_DIST:
        return False
        
    sorted_ys = sorted(p[1] for p in cluster)
    for j in range(len(sorted_ys) - 1):
        y_lo, y_hi = sorted_ys[j], sorted_ys[j + 1]
        if y_lo <= y_at_line <= y_hi:
            return True
    return False


# função para contar o número de linhas defensivas quebradas pelo passe
def _count_lines_broken(pass_x, pass_y, end_x, end_y, defenders_xy: list) -> int:
    lines = _cluster_defenders_into_lines(defenders_xy)
    n_broken = 0
    for line in lines:
        if _line_is_broken(pass_x, pass_y, end_x, end_y, line):
            n_broken += 1
    return n_broken


# contar adversários posicionados entre a origem e o destino do passe no eixo X
def _count_defenders_bypassed(pass_x, end_x, defenders_xy):
    return sum(1 for (dx, _) in defenders_xy if pass_x < dx < end_x)



# plot gráfico LBPV
def plot_pass_on_pitch(ax, row, events_df, title=''):

    draw_pitch(ax)

    px, py = float(row['pass_x']), float(row['pass_y'])
    ex, ey = float(row['end_x']),  float(row['end_y'])

    # Obter defesas do feeze frame
    mask = (
        (events_df['id'] == row['id']) &
        (events_df['teammate'] == False) &
        (events_df['actor']    == False) &
        (~events_df['keeper'].eq(True))
    )
    defenders_xy = [
        (float(np.asarray(r['location_y'], dtype=float)[0]),
         float(np.asarray(r['location_y'], dtype=float)[1]))
        for _, r in events_df[mask].iterrows()
        if r['location_y'] is not None
    ]

    # Desenhar cada linha defensiva
    for cluster in _cluster_defenders_by_x(defenders_xy):
        line_x    = np.mean([p[0] for p in cluster])
        in_path   = px <= line_x <= ex
        is_broken = in_path and _line_is_broken(px, py, ex, ey, cluster)
        is_single = len(cluster) < 2

        # As cores dos elementos do cluster são partilhadas entre si
        if is_broken:
            col, alpha_band, lw, ls = COL_BROKEN,  0.18, 2.2, '--'
        elif in_path:
            col, alpha_band, lw, ls = COL_IN_PATH, 0.10, 1.6, (0, (4, 2))
        else:
            col, alpha_band, lw, ls = COL_OUT,     0.06, 1.0, ':'

        ys = sorted(p[1] for p in cluster)
        y_lo, y_hi = ys[0], ys[-1]

        if not is_single:
            # Área de fundo transparente
            if in_path or line_x < px:
                xs   = [p[0] for p in cluster]
                span = max(xs) - min(xs) + 2.5
                ax.add_patch(plt.Rectangle(
                    (line_x - span / 2, 0), span, 80,
                    color=col, alpha=alpha_band, zorder=2,
                ))
            # Ligação vertical
            ax.plot(
                [line_x, line_x],
                [max(y_lo - 1.5, 0), min(y_hi + 1.5, 80)],
                color=col, linewidth=lw, linestyle=ls,
                alpha=0.85 if in_path else 0.45, zorder=3,
            )

        # Pontos dos defesas - cor herdada do cluster
        ax.scatter(
            [p[0] for p in cluster], [p[1] for p in cluster],
            s=70, color=col, edgecolors='white', linewidths=0.7,
            zorder=4, alpha=0.95,
        )

        # Ponto de intersecção
        if is_broken:
            ax.scatter(
                line_x, _pass_y_at_x(px, py, ex, ey, line_x),
                s=60, color=COL_BROKEN, marker='D',
                zorder=6, edgecolors='#374151', linewidths=1.2,
            )

    # Passador
    ax.scatter(px, py, s=130, color=COL_PASSER,
               zorder=7, marker='o', edgecolors='white', linewidth=1.5)

    # Direção do passe
    ax.annotate('', xy=(ex, ey), xytext=(px, py),
                 arrowprops=dict(arrowstyle='->', color=COL_PASS,
                                 lw=2.5, mutation_scale=20),
                 zorder=8)

    # Final do passe
    ax.scatter(ex, ey, s=80, color=COL_PASS, zorder=7, marker='x', linewidths=2)

    # Distância percorrida
    dist  = float(row.get('distance_advanced', ex - px))
    mid_x, mid_y = (px + ex) / 2, (py + ey) / 2
    ax.text(mid_x, mid_y + 2.8, f'{dist:.1f}m',
            color='#1e293b', fontsize=7.5, ha='center', va='bottom',
            fontweight='bold', zorder=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.7))

    ax.set_title(title, fontsize=9.5, color='#1e293b', pad=7, fontweight='bold')
