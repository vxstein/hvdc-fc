"""
graficos.py — Gráficos de áreas de vulnerabilidad y frecuencia de fallas
=========================================================================
Sigue el esquema de la memoria (Figuras 4.5 y 4.6):

  Fig. 4.5 izq: V vs ξ (perfil de tensión continuo por tramo)
  Fig. 4.5 der: fdp f(V) = |dξ/dV| (densidad de probabilidad de tensión)
  Fig. 4.6:     histograma fallas/año por bin de tensión (3φ y 1φ)

Tasas de falla:
  Líneas: LAMBDA_LINEA = 0.7 fallas/año por 100 km  (total)
  Barras: LAMBDA_BARRA = 0.08 fallas/año por barra   (total)
  F_3F = 0.05  fracción trifásicas
  F_1F = 0.80  fracción monofásicas
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    # Fuente tipo Times — el mismo aspecto que produce \usepackage{mathptmx}
    # en la memoria (TeX Gyre Termes / Liberation Serif son clones
    # métricamente compatibles con Times New Roman; se usa el primero
    # disponible en el sistema)
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Liberation Serif',
                           'Nimbus Roman', 'TeX Gyre Termes', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',   # más cercano a Times que 'cm' (Computer Modern)
    # Tamaños
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8,
    # Líneas y spines
    'axes.linewidth':    0.7,
    'lines.linewidth':   1.5,
    'patch.linewidth':   0.5,
    # Recuadro completo en los 4 lados, como el axis de pgfplots
    'axes.spines.top':   True,
    'axes.spines.right': True,
    # Grid sólido (equivalente a grid=major de pgfplots), no punteado
    'axes.grid':         True,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.55,
    'grid.linestyle':    '-',
    'grid.color':        '0.75',
    'axes.edgecolor':    'black',
    # Figura
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    # PDF vectorial con fuentes embebidas
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})
import matplotlib.pyplot as plt

from sic_datos import V_UMBRAL_3F, V_TRANS_1F_HI, V_TRANS_1F_LO

# Paleta alineada con la Figura de validación (tikz): colores primarios
# saturados en vez de la paleta pastel anterior, para que los gráficos del
# notebook luzcan consistentes con las figuras hechas a mano en LaTeX.
COLOR_1F_LINEA = 'tab:blue'
COLOR_3F_LINEA = 'tab:red'
COLOR_1F_BARRA = 'tab:green'
COLOR_3F_BARRA = 'tab:orange'
COLOR_UMBRAL   = 'purple'    # mismo color que la línea de "límite de tensión"

LAMBDA_LINEA  = 0.7
LAMBDA_BARRA  = 0.08
F_3F          = 0.05
F_1F          = 0.80
_FIG_DIR      = 'figuras_sic'

# Modelos de probabilidad de FC: V_UMBRAL_3F, V_TRANS_1F_HI, V_TRANS_1F_LO
# se importan desde sic_datos.py (fuente única de verdad, ec:pfc_trifasica / ec:pfc_monofasica)


def _pfc(v, tipo):
    """Probabilidad de FC dado tensión v y tipo de falla ('3f' o '1f')."""
    if tipo == '3f':
        return 1.0 if v < V_UMBRAL_3F else 0.0
    else:  # 1f
        if v >= V_TRANS_1F_HI:  return 0.0
        if v <= V_TRANS_1F_LO:  return 1.0
        return (V_TRANS_1F_HI - v) / (V_TRANS_1F_HI - V_TRANS_1F_LO)
N_BINS       = 20    # bins de 0.05 → 0.80, 0.85 y 0.90 caen exactamente en bordes


def _mostrar(fig, nombre_archivo):
    """Guarda la figura como PDF vectorial y la muestra en notebook si es posible."""
    nombre_pdf = nombre_archivo.replace('.png', '.pdf')
    os.makedirs(_FIG_DIR, exist_ok=True)
    ruta = os.path.join(_FIG_DIR, nombre_pdf)
    fig.savefig(ruta, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f"  Guardada: {ruta}")
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass
    plt.close(fig)


def _calcular_fdp(pts):
    """
    Calcula la fdp f(V) = |dξ/dV| numéricamente a partir de la curva V(ξ).

    Usa diferencias centrales para estimar dV/dξ en cada punto interior,
    y diferencias unilaterales en los extremos.

    Retorna (V_arr, fdp_arr) — arrays paralelos, ordenados por V creciente.
    """
    xi_arr = np.array([p[0] for p in pts])
    V_arr  = np.array([p[1] for p in pts])

    # dV/dξ por diferencias finitas
    dVdxi = np.gradient(V_arr, xi_arr)

    # |dξ/dV| = 1/|dV/dξ|, evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        fdp = np.where(np.abs(dVdxi) > 1e-9, 1.0 / np.abs(dVdxi), np.nan)

    # Ordenar por V para graficar correctamente
    orden  = np.argsort(V_arr)
    return V_arr[orden], fdp[orden]


def _pfc_integral(v_lo, v_hi, tipo):
    """
    Integral de P_FC(v) sobre el intervalo [v_lo, v_hi], dividida por (v_hi - v_lo).
    Equivale al valor medio de P_FC en el bin — exacto para funciones lineales por partes.
    """
    if v_hi <= v_lo:
        return 0.0
    dv = v_hi - v_lo

    if tipo == '3f':
        # Escalón en V_UMBRAL_3F
        if v_hi <= V_UMBRAL_3F:   return 1.0   # todo el bin con P_FC=1
        if v_lo >= V_UMBRAL_3F:   return 0.0   # todo el bin con P_FC=0
        # bin cruza el umbral
        return (V_UMBRAL_3F - v_lo) / dv

    else:  # 1f — lineal entre V_TRANS_1F_LO y V_TRANS_1F_HI
        hi, lo = V_TRANS_1F_HI, V_TRANS_1F_LO
        span = hi - lo

        def _pfc_val(v):
            if v >= hi:  return 0.0
            if v <= lo:  return 1.0
            return (hi - v) / span

        # Integral de P_FC sobre [v_lo, v_hi] usando trapecios exactos en los nodos
        # donde P_FC cambia de pendiente (v=lo y v=hi)
        nodes = sorted(set([v_lo, v_hi] + [x for x in [lo, hi] if v_lo < x < v_hi]))
        integral = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            integral += 0.5 * (_pfc_val(a) + _pfc_val(b)) * (b - a)
        return integral / dv


def _histograma_fallas_raw(pts, lambda_total, n_bins=N_BINS):
    V_arr  = np.array([p[1] for p in pts])
    V_arr  = V_arr[np.isfinite(V_arr)]
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    n      = len(V_arr)
    if n == 0:
        return bordes, counts
    for i in range(n_bins):
        frac = np.sum((V_arr >= bordes[i]) & (V_arr < bordes[i+1])) / n
        counts[i] = frac * lambda_total
    return bordes, counts


def _histograma_barras_raw(barras_crit, v_dict, lambda_por_barra, n_bins=N_BINS):
    """Histograma de hundimientos de tensión/año en barras (sin P_FC)."""
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    ancho  = bordes[1] - bordes[0]
    for k in barras_crit:
        v = v_dict[k]
        if not np.isfinite(v):
            continue
        bi = min(int(v / ancho), n_bins - 1)
        counts[bi] += lambda_por_barra
    return bordes, counts


def _histograma_fallas(pts, lambda_total, tipo, n_bins=N_BINS):
    """Histograma de FC/año por bin, ponderado por integral exacta de P_FC."""
    V_arr  = np.array([p[1] for p in pts])
    V_arr  = V_arr[np.isfinite(V_arr)]
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    n      = len(V_arr)
    if n == 0:
        return bordes, counts
    for i in range(n_bins):
        frac   = np.sum((V_arr >= bordes[i]) & (V_arr < bordes[i+1])) / n
        p_mean = _pfc_integral(bordes[i], bordes[i+1], tipo)
        counts[i] = frac * lambda_total * p_mean
    return bordes, counts


def _histograma_barras(barras_crit, v_dict, lambda_por_barra, tipo, n_bins=N_BINS):
    """
    Histograma de FC/año en barras por bin de tensión.
    Usa P_FC evaluada en el valor exacto de V_mk (punto, no bin).
    """
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    ancho  = bordes[1] - bordes[0]
    for k in barras_crit:
        v = v_dict[k]
        if not np.isfinite(v):
            continue
        bi = min(int(v / ancho), n_bins - 1)
        counts[bi] += lambda_por_barra * _pfc(v, tipo)
    return bordes, counts


def graficar_perfil_y_fdp(m_nombre, curvas, v_umbral):
    """
    Para cada corredor crítico genera una figura con 2×2 subplots:
      - Fila superior: falla 3φ
      - Fila inferior: falla 1φ
      - Col izquierda: V vs ξ
      - Col derecha:   fdp f(V)
    Replica la Figura 4.5 de la memoria.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    for c in criticas:
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
        partes = c['label'].split(' — ')
        titulo = (f"{partes[0].split()[0]}–{partes[1].split()[0]}"
                  if len(partes) == 2 else c['label'])
        fig.suptitle(f'{m_nombre}  |  Tramo {titulo}  (L={c["L_km"]:.0f} km)',
                     fontsize=12, fontweight='bold')

        for row, (tipo, pts, color) in enumerate([
            ('3φ', c['pts_3f'], COLOR_3F_LINEA),
            ('1φ', c['pts_1f'], COLOR_1F_LINEA),
        ]):
            xi_arr = np.array([p[0] for p in pts])
            V_arr  = np.array([p[1] for p in pts])
            # filtrar inf (puntos no críticos)
            mask   = np.isfinite(V_arr)
            xi_arr, V_arr = xi_arr[mask], V_arr[mask]
            if len(V_arr) == 0:
                continue
            V_fdp, fdp = _calcular_fdp(pts)

            # Col izquierda: V vs ξ
            ax = axes[row][0]
            ax.plot(xi_arr, V_arr, color=color, linewidth=1.8)
            ax.axhline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
                       linewidth=1, alpha=0.7)
            ax.fill_between(xi_arr, V_arr, 0,
                            where=V_arr <= v_umbral,
                            alpha=0.12, color=color)
            ax.set_xlabel('ξ (posición de falla)', fontsize=10)
            ax.set_ylabel(f'Tensión en {m_nombre.split()[0]} (p.u.)', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(V_arr.max() * 1.05, v_umbral * 1.1))
            ax.set_title(f'Falla {tipo}', fontsize=10)
            ax.grid(True, alpha=0.25)

            # Col derecha: fdp f(V)
            ax = axes[row][1]
            mask = np.isfinite(fdp)
            ax.plot(V_fdp[mask], fdp[mask], color=color, linewidth=1.8)
            ax.axvline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
                       linewidth=1, alpha=0.7)
            ax.set_xlabel(f'Tensión en {m_nombre.split()[0]} (p.u.)', fontsize=10)
            ax.set_ylabel('Densidad de probabilidad', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_title(f'fdp tensión ({tipo})', fontsize=10)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        nombre = f"{m_nombre.replace(' ','_')}_fig45_{titulo.replace('–','-')}.png"
        _mostrar(fig, nombre)


# ===========================================================================
# FIGURA 4.6 — Histograma fallas/año por bin de tensión
# ===========================================================================

def graficar_histograma_fallas(m_nombre, curvas, v_umbral):
    """
    Para cada corredor crítico genera un histograma de fallas/año por bin
    de tensión, con 3φ (rojo) y 1φ (azul) superpuestos.
    Replica la Figura 4.6 de la memoria.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    for c in criticas:
        L    = c['L_km']
        lam3 = F_3F * LAMBDA_LINEA * L / 100.0
        lam1 = F_1F * LAMBDA_LINEA * L / 100.0

        bordes3, counts3 = _histograma_fallas_raw(c['pts_3f'], lam3)
        bordes1, counts1 = _histograma_fallas_raw(c['pts_1f'], lam1)

        ancho   = bordes3[1] - bordes3[0]
        centros = 0.5 * (bordes3[:-1] + bordes3[1:])
        w       = ancho * 0.44   # mitad del ancho para barras adyacentes

        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        ax.bar(centros - w/2, counts1, width=w,
               color=COLOR_1F_LINEA, alpha=0.85, label=f'1φ  (λ={lam1:.4f} f/año)')
        ax.bar(centros + w/2, counts3, width=w,
               color=COLOR_3F_LINEA, alpha=0.85, label=f'3φ  (λ={lam3:.4f} f/año)')

        ax.axvline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
                   linewidth=1.2, alpha=0.8, label=f'Umbral {v_umbral} p.u.')

        ax.set_xlabel('Magnitud de tensión (p.u.)', fontsize=11)
        ax.set_ylabel('Hundimientos de tensión (por año)', fontsize=11)

        partes = c['label'].split(' — ')
        titulo = (f"{partes[0].split()[0]}–{partes[1].split()[0]}"
                  if len(partes) == 2 else c['label'])
        ax.set_title(f'{m_nombre}  |  Tramo {titulo}  (L={L:.0f} km)',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(bordes3[::2])
        ax.set_xticklabels([f'{v:.2f}' for v in bordes3[::2]], fontsize=7, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        nombre = f"{m_nombre.replace(' ','_')}_fig46_{titulo.replace('–','-')}.png"
        _mostrar(fig, nombre)


def graficar_frecuencia_acumulada_total(m_nombre, curvas, areas_b,
                                        v3f_barras, v1f_barras, v_umbral):
    """
    Histograma de fallas/año por bin de tensión, sumando todos los corredores
    críticos más las barras críticas (apiladas encima).
    Colores: azul=líneas 1φ, rojo=líneas 3φ, verde=barras 1φ, naranja=barras 3φ.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    bordes   = np.linspace(0.0, 1.0, N_BINS + 1)
    # Líneas
    lin_3f = np.zeros(N_BINS)
    lin_1f = np.zeros(N_BINS)
    for c in criticas:
        L    = c['L_km']
        _, counts3 = _histograma_fallas_raw(c['pts_3f'], F_3F * LAMBDA_LINEA * L / 100.0)
        _, counts1 = _histograma_fallas_raw(c['pts_1f'], F_1F * LAMBDA_LINEA * L / 100.0)
        lin_3f += counts3
        lin_1f += counts1

    # Barras
    lam_b3 = F_3F * LAMBDA_BARRA
    lam_b1 = F_1F * LAMBDA_BARRA
    _, bar_3f = _histograma_barras_raw(areas_b['3f'], v3f_barras, lam_b3)
    _, bar_1f = _histograma_barras_raw(areas_b['1f'], v1f_barras, lam_b1)

    ancho   = bordes[1] - bordes[0]
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    w       = ancho * 0.44

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # 1φ izquierda, 3φ derecha — adyacentes
    ax.bar(centros - w/2, lin_1f, width=w,
           color=COLOR_1F_LINEA, alpha=0.85,
           label=f'Líneas 1φ  (F={F_1F:.0%})')
    ax.bar(centros + w/2, lin_3f, width=w,
           color=COLOR_3F_LINEA, alpha=0.85,
           label=f'Líneas 3φ  (F={F_3F:.0%})')

    # Barras apiladas encima de líneas
    ax.bar(centros - w/2, bar_1f, width=w,
           bottom=lin_1f,
           color=COLOR_1F_BARRA, alpha=0.85,
           label=f'Barras 1φ')
    ax.bar(centros + w/2, bar_3f, width=w,
           bottom=lin_3f,
           color=COLOR_3F_BARRA, alpha=0.85,
           label=f'Barras 3φ')

    ax.axvline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
               linewidth=1.2, alpha=0.8, label=f'Umbral {v_umbral} p.u.')

    # Anotar total bajo el umbral
    idx_umb = int(round(v_umbral * N_BINS))
    total   = lin_1f + lin_3f + bar_1f + bar_3f
    f_umb   = total[:idx_umb].sum()
    ax.annotate(
        f'V < {v_umbral}: {f_umb:.4f} f/año',
        xy=(v_umbral, total[:idx_umb].max()), xycoords='data',
        xytext=(0.06, 0.80), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=8, color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.8)
    )

    ax.set_xlabel('Magnitud de tensión (p.u.)', fontsize=11)
    ax.set_ylabel('Hundimientos de tensión (por año)', fontsize=11)
    ax.set_title(
        f'{m_nombre} — Frecuencia total (líneas + barras)\n'
        f'({len(criticas)} corredores, {len(areas_b["1f"])} barras críticas 1φ, '
        f'{len(areas_b["3f"])} barras críticas 3φ)',
        fontsize=11, fontweight='bold'
    )
    ax.set_xticks(bordes[::2])
    ax.set_xticklabels([f'{v:.2f}' for v in bordes[::2]], fontsize=7, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)  # margen superior para que la anotación no choque con el título
    plt.tight_layout()
    nombre = f"{m_nombre.replace(' ','_')}_frecuencia_total.png"
    _mostrar(fig, nombre)


def graficar_cdf_total(m_nombre, curvas, areas_b,
                       v3f_barras, v1f_barras, v_umbral):
    """
    Gráfico de barras acumulado F(V) = fallas/año con tensión ≤ V.
    Incluye líneas y barras apiladas, con suma acumulativa de izquierda a derecha.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    bordes = np.linspace(0.0, 1.0, N_BINS + 1)
    lin_3f = np.zeros(N_BINS)
    lin_1f = np.zeros(N_BINS)

    for c in criticas:
        L = c['L_km']
        _, counts3 = _histograma_fallas_raw(c['pts_3f'], F_3F * LAMBDA_LINEA * L / 100.0)
        _, counts1 = _histograma_fallas_raw(c['pts_1f'], F_1F * LAMBDA_LINEA * L / 100.0)
        lin_3f += counts3
        lin_1f += counts1

    lam_b3 = F_3F * LAMBDA_BARRA
    lam_b1 = F_1F * LAMBDA_BARRA
    _, bar_3f = _histograma_barras_raw(areas_b['3f'], v3f_barras, lam_b3)
    _, bar_1f = _histograma_barras_raw(areas_b['1f'], v1f_barras, lam_b1)

    # Suma acumulativa — todos los bins
    cdf_lin_1f = np.cumsum(lin_1f)
    cdf_lin_3f = np.cumsum(lin_3f)
    cdf_bar_1f = np.cumsum(bar_1f)
    cdf_bar_3f = np.cumsum(bar_3f)
    cdf_total  = cdf_lin_1f + cdf_lin_3f + cdf_bar_1f + cdf_bar_3f

    ancho   = bordes[1] - bordes[0]
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    bordes_plot = bordes

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Líneas (base)
    ax.bar(centros, cdf_lin_1f, width=ancho * 0.95,
           color=COLOR_1F_LINEA, alpha=0.85, label=f'Líneas 1φ  (F={F_1F:.0%})')
    ax.bar(centros, cdf_lin_3f, width=ancho * 0.95,
           color=COLOR_3F_LINEA, alpha=0.75, label=f'Líneas 3φ  (F={F_3F:.0%})')

    # Barras (apiladas)
    ax.bar(centros, cdf_bar_1f, width=ancho * 0.95,
           bottom=cdf_lin_1f,
           color=COLOR_1F_BARRA, alpha=0.85, label=f'Barras 1φ  (F={F_1F:.0%})')
    ax.bar(centros, cdf_bar_3f, width=ancho * 0.95,
           bottom=cdf_lin_3f,
           color=COLOR_3F_BARRA, alpha=0.75, label=f'Barras 3φ  (F={F_3F:.0%})')

    ax.axvline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
               linewidth=1.2, alpha=0.8, label=f'Umbral {v_umbral} p.u.')

    # Anotar valor en el umbral
    idx_umb = int(round(v_umbral / ancho)) - 1
    idx_umb = min(idx_umb, N_BINS - 1)
    f_umb   = cdf_total[idx_umb]
    ax.annotate(
        f'F({v_umbral}) = {f_umb:.4f} f/año',
        xy=(centros[idx_umb], cdf_total[idx_umb]), xycoords='data',
        xytext=(0.06, 0.80), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.8)
    )

    ax.set_xlabel('Magnitud de tensión V (p.u.)', fontsize=11)
    ax.set_ylabel('Hundimientos acumulados (por año)', fontsize=11)
    ax.set_title(
        f'{m_nombre} — F(V): fallas/año con tensión ≤ V\n'
        f'(líneas + barras)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(bordes_plot[::2])
    ax.set_xticklabels([f'{v:.2f}' for v in bordes_plot[::2]], fontsize=7, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_xlim(0, bordes_plot[-1])
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)  # margen superior para que la anotación no choque con el título
    plt.tight_layout()
    nombre = f"{m_nombre.replace(' ','_')}_cdf_total.png"
    _mostrar(fig, nombre)

def graficar_fc_acumulada(m_nombre, curvas, areas_b,
                          v3f_barras, v1f_barras, v_umbral):
    """
    Histograma acumulado de FC/año (ponderado por P_FC).
    Gráfico final por barra de observación.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    bordes = np.linspace(0.0, 1.0, N_BINS + 1)
    lin_3f = np.zeros(N_BINS)
    lin_1f = np.zeros(N_BINS)

    for c in criticas:
        L = c['L_km']
        _, cnt3 = _histograma_fallas(c['pts_3f'], F_3F * LAMBDA_LINEA * L / 100.0, '3f')
        _, cnt1 = _histograma_fallas(c['pts_1f'], F_1F * LAMBDA_LINEA * L / 100.0, '1f')
        lin_3f += cnt3
        lin_1f += cnt1

    _, bar_3f = _histograma_barras(areas_b['3f'], v3f_barras, F_3F * LAMBDA_BARRA, '3f')
    _, bar_1f = _histograma_barras(areas_b['1f'], v1f_barras, F_1F * LAMBDA_BARRA, '1f')

    cdf_lin_1f = np.cumsum(lin_1f)
    cdf_lin_3f = np.cumsum(lin_3f)
    cdf_bar_1f = np.cumsum(bar_1f)
    cdf_bar_3f = np.cumsum(bar_3f)

    # No dibujar datos en los bins con V >= v_umbral, pero mantenerlos visibles
    idx_desde_umbral = int(round(v_umbral * N_BINS))
    for arr in [cdf_lin_1f, cdf_lin_3f, cdf_bar_1f, cdf_bar_3f]:
        arr[idx_desde_umbral:] = 0

    ancho   = bordes[1] - bordes[0]
    centros = 0.5 * (bordes[:-1] + bordes[1:])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Apilado: líneas 1φ (base) → barras 1φ → líneas 3φ → barras 3φ
    ax.bar(centros, cdf_lin_1f, width=ancho * 0.95,
           color=COLOR_1F_LINEA, alpha=0.85, label=f'Líneas 1φ')
    ax.bar(centros, cdf_bar_1f, width=ancho * 0.95,
           bottom=cdf_lin_1f,
           color=COLOR_1F_BARRA, alpha=0.85, label=f'Barras 1φ')
    ax.bar(centros, cdf_lin_3f, width=ancho * 0.95,
           bottom=cdf_lin_1f + cdf_bar_1f,
           color=COLOR_3F_LINEA, alpha=0.85, label=f'Líneas 3φ')
    ax.bar(centros, cdf_bar_3f, width=ancho * 0.95,
           bottom=cdf_lin_1f + cdf_bar_1f + cdf_lin_3f,
           color=COLOR_3F_BARRA, alpha=0.85, label=f'Barras 3φ')
    ax.axvline(v_umbral, color=COLOR_UMBRAL, linestyle='--',
               linewidth=1.2, alpha=0.8, label=f'Umbral {v_umbral} p.u.')

    # Anotar tasa total de FC en el umbral (bin inmediatamente anterior a v_umbral)
    idx_umb = int(round(v_umbral * N_BINS)) - 1
    idx_umb = max(0, min(idx_umb, N_BINS - 1))
    f_fc    = (cdf_lin_1f + cdf_bar_1f + cdf_lin_3f + cdf_bar_3f)[idx_umb]
    f_total_plot = cdf_lin_1f + cdf_bar_1f + cdf_lin_3f + cdf_bar_3f
    ax.annotate(
        f'FC/año = {f_fc:.4f}',
        xy=(centros[idx_umb], f_total_plot[idx_umb]), xycoords='data',
        xytext=(0.06, 0.80), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='#2c3e50', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.8)
    )

    ax.set_xlabel('Magnitud de tensión V (p.u.)', fontsize=11)
    ax.set_ylabel('FC acumuladas [FC/año]', fontsize=11)
    ax.set_title(
        f'{m_nombre} — Tasa anual de fallas de conmutación\n'
        f'({len(criticas)} corredores + {len(areas_b["1f"])} barras críticas)',
        fontsize=11, fontweight='bold'
    )
    ax.set_xticks(bordes[::2])
    ax.set_xticklabels([f'{v:.2f}' for v in bordes[::2]], fontsize=7, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)  # margen superior para que la anotación no choque con el título
    plt.tight_layout()
    nombre = f"{m_nombre.replace(' ','_')}_fc_acumulada.png"
    _mostrar(fig, nombre)



def calcular_tasa_fc(curvas, areas_b, v3f_barras, v1f_barras, v_umbral):
    """
    Calcula la tasa anual de FC (líneas + barras, 3φ + 1φ) para una barra
    de observación. Replica exactamente la integración usada en
    graficar_fc_acumulada, de modo que el valor coincida con el anotado
    en la figura.

    Retorna un dict con el desglose por origen/tipo de falla y el total.
    """
    labels_vistos = set()
    criticas = []
    for c in curvas:
        if c['label'] not in labels_vistos and (c['critico_3f'] or c['critico_1f']):
            criticas.append(c)
            labels_vistos.add(c['label'])

    lin_3f = np.zeros(N_BINS)
    lin_1f = np.zeros(N_BINS)
    for c in criticas:
        L = c['L_km']
        _, cnt3 = _histograma_fallas(c['pts_3f'], F_3F * LAMBDA_LINEA * L / 100.0, '3f')
        _, cnt1 = _histograma_fallas(c['pts_1f'], F_1F * LAMBDA_LINEA * L / 100.0, '1f')
        lin_3f += cnt3
        lin_1f += cnt1

    _, bar_3f = _histograma_barras(areas_b['3f'], v3f_barras, F_3F * LAMBDA_BARRA, '3f')
    _, bar_1f = _histograma_barras(areas_b['1f'], v1f_barras, F_1F * LAMBDA_BARRA, '1f')

    cdf_lin_3f = np.cumsum(lin_3f)
    cdf_lin_1f = np.cumsum(lin_1f)
    cdf_bar_3f = np.cumsum(bar_3f)
    cdf_bar_1f = np.cumsum(bar_1f)

    idx_umb = int(round(v_umbral * N_BINS)) - 1
    idx_umb = max(0, min(idx_umb, N_BINS - 1))

    return dict(
        lineas_3f=cdf_lin_3f[idx_umb],
        lineas_1f=cdf_lin_1f[idx_umb],
        barras_3f=cdf_bar_3f[idx_umb],
        barras_1f=cdf_bar_1f[idx_umb],
        total=(cdf_lin_3f + cdf_lin_1f + cdf_bar_3f + cdf_bar_1f)[idx_umb],
    )


def imprimir_tasas_fc(curvas_obs, areas_b, v3f_barras, v1f_barras,
                       obs_buses, v_umbral):
    """
    Resumen en texto de la tasa anual de fallas de conmutación (FC/año)
    por barra de observación, con desglose por origen (líneas/barras) y
    tipo de falla (3φ/1φ). Usa la misma integración que graficar_fc_acumulada,
    por lo que el total coincide con la anotación de esa figura.
    """
    ANCHO = 60
    print(f"\n{'='*ANCHO}")
    print("  TASA ANUAL DE FALLAS DE CONMUTACIÓN (FC/año)")
    print('='*ANCHO)
    for m, m_nombre in obs_buses.items():
        r = calcular_tasa_fc(curvas_obs[m], areas_b[m],
                              v3f_barras[m], v1f_barras[m], v_umbral)
        print(f"\n  ► {m_nombre}  (barra {m})")
        print(f"    Líneas   3φ: {r['lineas_3f']:.4f}   1φ: {r['lineas_1f']:.4f}")
        print(f"    Barras   3φ: {r['barras_3f']:.4f}   1φ: {r['barras_1f']:.4f}")
        print(f"    {'-'*38}")
        print(f"    TOTAL FC/año: {r['total']:.4f}")
    print(f"\n{'='*ANCHO}")


def generar_graficos(curvas_obs, areas_b, areas_l,
                     v3f_barras, v1f_barras,
                     obs_buses, nombres, v_umbral):
    """
    Genera Figuras 4.5 y 4.6 de la memoria para cada barra de observación,
    más un gráfico final de frecuencia acumulada total por barra.
    """
    for m, m_nombre in obs_buses.items():
        curvas = curvas_obs[m]
        n_crit = len([c for c in curvas if c['critico_3f'] or c['critico_1f']])
        print(f"\n{'='*65}")
        print(f"  GRÁFICOS — {m_nombre}  ({n_crit} corredores críticos)")
        print(f"{'='*65}")

        print(f"  → Figura 4.5: perfil V(ξ) y fdp por corredor")
        graficar_perfil_y_fdp(m_nombre, curvas, v_umbral)

        print(f"  → Figura 4.6: histograma fallas/año por corredor")
        graficar_histograma_fallas(m_nombre, curvas, v_umbral)

        print(f"  → Frecuencia acumulada total")
        graficar_frecuencia_acumulada_total(
            m_nombre, curvas, areas_b[m],
            v3f_barras[m], v1f_barras[m], v_umbral
        )

        print(f"  → CDF: hundimientos acumulados")
        graficar_cdf_total(
            m_nombre, curvas, areas_b[m],
            v3f_barras[m], v1f_barras[m], v_umbral
        )

        print(f"  → FC acumuladas: tasa anual de FC")
        graficar_fc_acumulada(
            m_nombre, curvas, areas_b[m],
            v3f_barras[m], v1f_barras[m], v_umbral
        )


if __name__ == '__main__':
    print("Importar desde main_sic.py para generar gráficos.")
