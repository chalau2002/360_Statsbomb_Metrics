import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mplsoccer import Pitch
from scipy.spatial import ConvexHull, Voronoi
import matplotlib.path as mpath
import matplotlib.patches as patches

# Constants para os clusters e linha defensiva
LINE_GAP_THRESHOLD = 4.5    # gap em X > 4.5m → nova linha defensiva
MIN_LINE_SIZE      = 2      # uma linha necessita de pelo menos 2 defesas
MIN_START_DIST     = 5.0    # o passador deve estar a pelo menos 5m do ponto de interseção


# Palete de cores
COL_BROKEN  = '#facc15'  # amarelo — linha quebrada / respetivos defesas
COL_IN_PATH = '#f97316'  # laranja — linha no trajeto mas não quebrada
COL_OUT     = '#64748b'  # cinzento — linha fora do corredor do passe
COL_PASSER  = '#22c55e'  # verde — passador
COL_PASS    = '#22d3ee'  # azul claro — trajetória do passe



# Grupos de posições para consolidação e filtragem de jogadores
POSITION_GROUPS = {
    'Guarda-redes': ['Goalkeeper'],
    'Defesas Centrais': ['Center Back', 'Left Center Back', 'Right Center Back'],
    'Laterais': ['Left Back', 'Right Back', 'Left Wing Back', 'Right Wing Back'],
    'Médios': [
        'Center Defensive Midfield', 'Left Defensive Midfield', 'Right Defensive Midfield',
        'Left Center Midfield', 'Right Center Midfield', 'Center Attacking Midfield'
    ],
    'Extremos': [
        'Left Attacking Midfield', 'Right Attacking Midfield',
        'Left Midfield', 'Right Midfield', 'Left Wing', 'Right Wing'
    ],
    'Avançados': ['Center Forward', 'Left Center Forward', 'Right Center Forward'],
    'Outros/Substitutos': ['Substitute', 'Desconhecido']
}

# Dicionário posição individual -> grupo de posição
POS_TO_GROUP = {pos: grp for grp, positions in POSITION_GROUPS.items() for pos in positions}

def draw_pitch(ax, bg='white', line_color='#374151'):
    Pitch(
        pitch_type='statsbomb',
        pitch_color=bg,
        line_color=line_color,
        linewidth=1.5,
        line_alpha=0.85,
        goal_type='box',
    ).draw(ax=ax)


# Helpers para legenda
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


# Title helper LBPV
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
        f"{player} ({team[:3].upper()})  LBPV: {row['score']:.2f}\n"
        f"Min {minute}  |  {lines} line(s)  |  {defs} def bypassed"
        f"  |  {dist:.1f}m  |  {outcome}"
        f"{match_str}"
    )



# abreviar nome do jogador para "P. Último"
def _get_short_name(name):
    if name is None or (isinstance(name, float) and np.isnan(name)) or not isinstance(name, str) or name.strip() == '':
        return None
    parts = name.split()
    if len(parts) > 1:
        return f"{parts[0][0]}. {parts[-1]}"
    return name


# funçao para extrair coordenadas (x, y) de um array/lista de forma robusta
def _extract_xy(arr):
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.nan, np.nan
    if isinstance(arr, (list, tuple, np.ndarray)):
        if len(arr) >= 2:
            try:
                return float(arr[0]), float(arr[1])
            except (ValueError, TypeError):
                return np.nan, np.nan
        return np.nan, np.nan
    if isinstance(arr, str):
        cleaned = arr.strip('{}[]() \t\n\r')
        if not cleaned:
            return np.nan, np.nan
        parts = cleaned.split(',')
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except (ValueError, TypeError):
                return np.nan, np.nan
    return np.nan, np.nan


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
def plot_pass_on_pitch(ax, row, events_df, title=None, matches_df=None):

    if title is None:
        title = pass_title(row, matches_df)

    draw_pitch(ax)

    if pd.isna(row.get('pass_x')) or pd.isna(row.get('pass_y')) or pd.isna(row.get('end_x')) or pd.isna(row.get('end_y')):
        ax.text(60, 40, "Coordenadas de passe em falta.", ha='center', va='center', color='red', fontsize=12, fontweight='bold')
        return

    px, py = float(row['pass_x']), float(row['pass_y'])
    ex, ey = float(row['end_x']),  float(row['end_y'])

    # Obter defesas do freeze frame
    mask = (
        (events_df['id'] == row['id']) &
        (events_df['teammate'] == False) &
        (events_df['actor']    == False) &
        (~events_df['keeper'].eq(True))
    )
    defenders_xy = []
    for _, r in events_df[mask].iterrows():
        coord = _extract_xy(r['location_y'])
        if not np.isnan(coord[0]) and not np.isnan(coord[1]):
            defenders_xy.append(coord)

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

    # Identificar passador no gráfico
    passer_short = _get_short_name(row.get('player'))
    if passer_short:
        passer_y = py - 3.0 if py > 8 else py + 3.0
        ax.text(px, passer_y, passer_short,
                color='#166534', fontsize=6.0, ha='center', va='center',
                fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=COL_PASSER, alpha=0.85, linewidth=0.7))

    # Direção do passe
    ax.annotate('', xy=(ex, ey), xytext=(px, py),
                 arrowprops=dict(arrowstyle='->', color=COL_PASS,
                                 lw=2.5, mutation_scale=20),
                 zorder=8)

    # Final do passe
    ax.scatter(ex, ey, s=80, color=COL_PASS, zorder=7, marker='x', linewidths=2)

    # Identificar receptor no gráfico
    receiver_short = _get_short_name(row.get('pass_recipient'))
    if receiver_short:
        receiver_y = ey + 3.0 if ey < 72 else ey - 3.0
        ax.text(ex, receiver_y, receiver_short,
                color='#0e7490', fontsize=6.0, ha='center', va='center',
                fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=COL_PASS, alpha=0.85, linewidth=0.7))

    # Distância percorrida
    dist  = float(row.get('distance_advanced', ex - px))
    mid_x, mid_y = (px + ex) / 2, (py + ey) / 2
    ax.text(mid_x, mid_y + 2.8, f'{dist:.1f}m',
            color='#1e293b', fontsize=7.5, ha='center', va='bottom',
            fontweight='bold', zorder=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.7))

    ax.set_title(title, fontsize=9.5, color='#1e293b', pad=7, fontweight='bold')


# Helpers para RAI

# helper para o titulo do RAI
def reception_title(row, matches_df=None):
    player  = row.get('player', 'Unknown')
    team    = row.get('team', '')
    minute  = int(row.get('minute', 0))
    
    score   = float(row.get('score', 0.0))
    space   = float(row.get('voronoi_area', 0.0))
    dens    = float(row.get('defensive_density', 0.0))
    dist    = float(row.get('nearest_defender_distance', 0.0))
    diff    = float(row.get('difficulty_context', 0.0))

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
        f"{player} ({team[:3].upper()})  RAI: {score:.2f}\n"
        f"Min {minute}  |  Space: {space:.1f}m²  |  Density: {dens:.1f}"
        f"  |  Nearest Def.: {dist:.1f}m  |  Difficulty: {diff:.3f}"
        f"{match_str}"
    )


# Helpers para a legenda do RAI
def reception_legend_handles():
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    return [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#15803d',
                      markeredgecolor='white', markersize=9, label='Receiver'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ade80',
                      markeredgecolor='white', markersize=7, label='Teammates'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f87171',
                      markeredgecolor='white', markersize=7, label='Defenders'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#fbbf24',
                      markeredgecolor='black', markersize=8, label='Goalkeeper'),
        mlines.Line2D([0], [0], marker='D', color='w', markerfacecolor='#ef4444',
                      markeredgecolor='black', markersize=7, label='Nearest Defender'),
        mpatches.Patch(facecolor='#22c55e', edgecolor='#15803d', alpha=0.30,
                       linewidth=1.5, label='Voronoi inside Hull (Controlled Space)'),
        mpatches.Patch(facecolor='#ef4444', edgecolor='#dc2626', alpha=0.15,
                       linewidth=1.5, linestyle='--', label='Defensive Convex Hull'),
        mlines.Line2D([0], [0], color='#b91c1c', linewidth=1.2, linestyle=':',
                      label='Density / Proximity (3m)')
    ]

# Adicionar legenda do RAI na parte inferior central da figura.
def add_reception_legend(fig, ncol=4, fontsize=9, anchor=(0.5, 0.0)):
    fig.legend(
        handles=reception_legend_handles(),
        loc='lower center',
        ncol=ncol,
        fontsize=fontsize,
        framealpha=0.0,
        edgecolor='none',
        bbox_to_anchor=anchor,
    )


# função para verificar se o recetor está dentro do convex hull dos defesas
def _is_inside_convex_hull(point, hull_points) -> bool:
    
    # Verificar se o número de defesas é suficiente para formar um hull
    if len(hull_points) < 3:
        return False
    try:
        hull_pts = np.asarray(hull_points)
        hull = ConvexHull(hull_pts)
        hull_path = mpath.Path(hull_pts[hull.vertices])
        # verificar se o ponto está dentro ou na fronteira do hull
        return bool(hull_path.contains_point(point) or hull_path.contains_point(point, radius=1e-5))
    except Exception:
        return False


# função para calcular a área do convex hull defensivo
def _get_convex_hull_area(points) -> float:
    # Verificar se o número de pontos é suficiente para formar um hull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


# função para calcular a interseção de duas linhas
def _get_line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return None
        
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return [x, y]
    return None


# função para calcular a área de interseção entre dois polígonos convexos
def _intersect_convex_polygons(poly1, poly2) -> float:
    p1 = np.asarray(poly1, dtype=float)
    p2 = np.asarray(poly2, dtype=float)
    path1 = mpath.Path(p1)
    path2 = mpath.Path(p2)
    
    pts = []
    # Vértices de poly1 dentro de poly2
    for pt in p1:
        if path2.contains_point(pt) or path2.contains_point(pt, radius=1e-5):
            pts.append(pt)
    # Vértices de poly2 dentro de poly1
    for pt in p2:
        if path1.contains_point(pt) or path1.contains_point(pt, radius=1e-5):
            pts.append(pt)
    # Interseções entre as arestas
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        edge1_p1 = p1[i]
        edge1_p2 = p1[(i+1)%n1]
        for j in range(n2):
            edge2_p1 = p2[j]
            edge2_p2 = p2[(j+1)%n2]
            inter = _get_line_intersection(edge1_p1, edge1_p2, edge2_p1, edge2_p2)
            if inter is not None:
                pts.append(inter)
                
    if len(pts) < 3:
        return 0.0
    try:
        pts = np.unique(np.array(pts).round(6), axis=0)
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 0.0


# função para obter os vértices da célula Voronoi de forma ordenada e limitada ao campo
def _get_clipped_voronoi_vertices(all_points, target_point, pitch_length=120.0, pitch_width=80.0):
    all_pts = np.asarray(all_points, dtype=float)
    target = np.asarray(target_point, dtype=float)
    if len(all_pts) <= 1:
        return np.array([[0.0, 0.0], [pitch_length, 0.0], [pitch_length, pitch_width], [0.0, pitch_width]])
        
    reflected = []
    for pt in all_pts:
        x, y = pt[0], pt[1]
        reflected.append([-x, y])
        reflected.append([2 * pitch_length - x, y])
        reflected.append([x, -y])
        reflected.append([x, 2 * pitch_width - y])
        reflected.append([-x, -y])
        reflected.append([-x, 2 * pitch_width - y])
        reflected.append([2 * pitch_length - x, -y])
        reflected.append([2 * pitch_length - x, 2 * pitch_width - y])
        
    expanded = np.vstack([all_pts, reflected])
    try:
        vor = Voronoi(expanded)
        dists_real = np.sum((all_pts - target) ** 2, axis=1)
        target_idx = np.argmin(dists_real)
        
        region_idx = vor.point_region[target_idx]
        vertices_idxs = vor.regions[region_idx]
        if -1 in vertices_idxs or not vertices_idxs:
            return np.array([])
        vertices = vor.vertices[vertices_idxs]
        
        # Ordenar os vértices em sentido anti-horário
        hull = ConvexHull(vertices)
        return vertices[hull.vertices]
    except Exception:
        return np.array([])


# função para calcular o Voronoi no bloco defensivo
def _get_voronoi_area_in_hull(all_points, target_point, hull_points, pitch_length=120.0, pitch_width=80.0) -> float:
    if len(hull_points) < 3:
        return 0.0
        
    vor_vertices = _get_clipped_voronoi_vertices(all_points, target_point, pitch_length, pitch_width)
    if len(vor_vertices) == 0:
        return 0.0
        
    try:
        hull_pts = np.asarray(hull_points)
        hull = ConvexHull(hull_pts)
        hull_vertices = hull_pts[hull.vertices]
        return _intersect_convex_polygons(vor_vertices, hull_vertices)
    except Exception:
        return 0.0


# função para calcular a área Voronoi
def _get_clipped_voronoi_area(all_points, target_point, hull_points=None, pitch_length=120.0, pitch_width=80.0) -> float:
    if hull_points is not None and len(hull_points) >= 3:
        return _get_voronoi_area_in_hull(all_points, target_point, hull_points, pitch_length, pitch_width)
        
    # Fallback to standard Voronoi
    all_pts = np.asarray(all_points, dtype=float)
    target = np.asarray(target_point, dtype=float)
    if len(all_pts) <= 1:
        return pitch_length * pitch_width
        
    reflected = []
    for pt in all_pts:
        x, y = pt[0], pt[1]
        reflected.append([-x, y])
        reflected.append([2 * pitch_length - x, y])
        reflected.append([x, -y])
        reflected.append([x, 2 * pitch_width - y])
        reflected.append([-x, -y])
        reflected.append([-x, 2 * pitch_width - y])
        reflected.append([2 * pitch_length - x, -y])
        reflected.append([2 * pitch_length - x, 2 * pitch_width - y])
        
    expanded = np.vstack([all_pts, reflected])
    try:
        vor = Voronoi(expanded)
        dists_real = np.sum((all_pts - target) ** 2, axis=1)
        target_idx = np.argmin(dists_real)
        
        region_idx = vor.point_region[target_idx]
        vertices_idxs = vor.regions[region_idx]
        if -1 in vertices_idxs or not vertices_idxs:
            return 0.0
        vertices = vor.vertices[vertices_idxs]
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        return 0.0


# função para calcular a densidade defensiva
def _get_defensive_density(target_point, defender_points, radius=3.0) -> int:
    if not defender_points:
        return 0
    target = np.asarray(target_point, dtype=float)
    defs = np.asarray(defender_points, dtype=float)
    dists = np.sqrt(np.sum((defs - target) ** 2, axis=1))
    return int(np.sum(dists <= radius))

# função para calcular a distância ao defensor mais próximo
def _get_nearest_defender_distance(target_point, defender_points) -> float:
    if not defender_points:
        return 50.0
    target = np.asarray(target_point, dtype=float)
    defs = np.asarray(defender_points, dtype=float)
    dists = np.sqrt(np.sum((defs - target) ** 2, axis=1))
    return float(np.min(dists))


# função para plotar uma receção de bola no campo
def plot_reception_on_pitch(ax, row, events_df, title=None, matches_df=None):
    
    if title is None:
        title = reception_title(row, matches_df)
        
    # Desenhar o campo
    draw_pitch(ax)
    
    if pd.isna(row.get('rec_x')) or pd.isna(row.get('rec_y')):
        ax.text(60, 40, "Coordenadas de receção em falta.", ha='center', va='center', color='red', fontsize=12, fontweight='bold')
        return

    rx, ry = float(row['rec_x']), float(row['rec_y'])
    
    # obter jogadores do freeze frame para este evento
    mask = (events_df['id'] == row['id']) & (events_df['location_y'].notna())
    frame_players = events_df[mask].copy()
    if frame_players.empty:
        # Se não houver freeze frame, apenas desenha o recetor
        ax.scatter(rx, ry, s=140, color='#15803d', edgecolors='white', linewidths=1.5, zorder=6)
        rec_short = _get_short_name(row.get('player', 'Recetor'))
        if rec_short:
            rec_y = ry - 3.5 if ry > 8 else ry + 3.5
            ax.text(rx, rec_y, rec_short, color='#14532d', fontsize=7.0, ha='center', va='center', fontweight='bold', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#22c55e', alpha=0.9, linewidth=0.8))
        ax.set_title(title, fontsize=10.0, color='#1e293b', pad=8, fontweight='bold')
        return

    # Extrai e filtra apenas coordenadas válidas
    frame_players[['px', 'py']] = frame_players['location_y'].apply(lambda v: pd.Series(_extract_xy(v)))
    frame_players = frame_players.dropna(subset=['px', 'py'])
    
    # separar em recetor, colegas, adversários e guarda-redes
    opponents_df = frame_players[(frame_players['teammate'] == False) & (~frame_players['keeper'].eq(True))]
    teammates_df = frame_players[(frame_players['teammate'] == True) & (frame_players['actor'] == False)]
    keeper_df = frame_players[frame_players['keeper'] == True]
    
    opponents_xy = list(zip(opponents_df['px'], opponents_df['py']))
    all_players_xy = list(zip(frame_players['px'], frame_players['py']))
    
    # desenhar Convex Hull dos adversários
    if len(opponents_xy) >= 3:
        try:
            hull_arr = np.array(opponents_xy)
            hull = ConvexHull(hull_arr)
            hull_polygon = hull_arr[hull.vertices]
            poly = patches.Polygon(hull_polygon, closed=True, facecolor='#ef4444', alpha=0.15, edgecolor='#dc2626', linewidth=1.5, linestyle='--', zorder=2)
            ax.add_patch(poly)
        except Exception:
            pass
            
    # desenhar Célula Voronoi do recetor dentro do bloco (interseção)
    if len(all_players_xy) > 1 and len(opponents_xy) >= 3:
        try:
            vor_vertices = _get_clipped_voronoi_vertices(all_players_xy, (rx, ry))
            hull_arr = np.array(opponents_xy)
            hull = ConvexHull(hull_arr)
            hull_vertices = hull_arr[hull.vertices]
            
            # obter os pontos da interseção de ambos os polígonos
            p1 = np.asarray(vor_vertices)
            p2 = np.asarray(hull_vertices)
            path1 = mpath.Path(p1)
            path2 = mpath.Path(p2)
            
            pts = []
            for pt in p1:
                if path2.contains_point(pt) or path2.contains_point(pt, radius=1e-5):
                    pts.append(pt)
            for pt in p2:
                if path1.contains_point(pt) or path1.contains_point(pt, radius=1e-5):
                    pts.append(pt)
            n1, n2 = len(p1), len(p2)
            for i in range(n1):
                edge1_p1 = p1[i]
                edge1_p2 = p1[(i+1)%n1]
                for j in range(n2):
                    edge2_p1 = p2[j]
                    edge2_p2 = p2[(j+1)%n2]
                    inter = _get_line_intersection(edge1_p1, edge1_p2, edge2_p1, edge2_p2)
                    if inter is not None:
                        pts.append(inter)
            
            if len(pts) >= 3:
                pts = np.unique(np.array(pts).round(6), axis=0)
                v_hull = ConvexHull(pts)
                v_polygon = pts[v_hull.vertices]
                v_poly = patches.Polygon(v_polygon, closed=True, facecolor='#22c55e', alpha=0.30, edgecolor='#15803d', linewidth=1.5, zorder=1)
                ax.add_patch(v_poly)
        except Exception:
            pass
            
    # desenhar círculo de raio 3m ao redor do recetor
    circle = patches.Circle((rx, ry), radius=3.0, facecolor='none', edgecolor='#b91c1c', linewidth=1.0, linestyle=':', alpha=0.6, zorder=3)
    ax.add_patch(circle)
    
    # desenhar jogadores
    # colegas de equipa (verde)
    if not teammates_df.empty:
        ax.scatter(teammates_df['px'], teammates_df['py'], s=60, color='#4ade80', edgecolors='white', linewidths=0.7, alpha=0.85, zorder=4)
    # adversários (vermelho)
    if not opponents_df.empty:
        ax.scatter(opponents_df['px'], opponents_df['py'], s=65, color='#f87171', edgecolors='white', linewidths=0.7, alpha=0.9, zorder=4)
    # guarda-redes (amarelo)
    if not keeper_df.empty:
        ax.scatter(keeper_df['px'], keeper_df['py'], s=75, color='#fbbf24', edgecolors='black', linewidths=1.0, alpha=0.95, zorder=5)
        
    # desenhar recetor (verde escuro)
    ax.scatter(rx, ry, s=140, color='#15803d', edgecolors='white', linewidths=1.5, zorder=6)
    
    # rótulo de texto para o recetor
    rec_short = _get_short_name(row.get('player', 'Recetor'))
    if rec_short:
        rec_y = ry - 3.5 if ry > 8 else ry + 3.5
        ax.text(rx, rec_y, rec_short, color='#14532d', fontsize=7.0, ha='center', va='center', fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#22c55e', alpha=0.9, linewidth=0.8))
                
    # destacar defesa mais próximo (linha tracejada vermelha)
    if not opponents_df.empty:
        opps_arr = np.array(opponents_xy)
        dists = np.sqrt(np.sum((opps_arr - np.array([rx, ry])) ** 2, axis=1))
        min_idx = np.argmin(dists)
        near_x, near_y = opps_arr[min_idx]
        ax.plot([rx, near_x], [ry, near_y], color='#b91c1c', linewidth=1.5, linestyle=':', alpha=0.8, zorder=3)
        # rótulo do defesa mais próximo
        ax.scatter(near_x, near_y, s=80, color='#ef4444', marker='D', edgecolors='black', linewidths=0.8, zorder=5)
        
    ax.set_title(title, fontsize=10.0, color='#1e293b', pad=8, fontweight='bold')

