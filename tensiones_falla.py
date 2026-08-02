"""
tensiones_falla.py — Tensiones de Falla en Barras y Tramos de Línea
====================================================================
Calcula la tensión en barras de observación (Cardones, Alto Jahuel,
Valdivia) ante fallas de cortocircuito en cualquier punto del sistema.

Ecuaciones de la memoria (VR, 2026):

FALLA TRIFÁSICA EN BARRA k — Ec. 3.8:
  V_mk = 1 − Z1_mk / Z1_kk

FALLA MONOFÁSICA EN BARRA k — Ec. 3.13 (fórmula exacta de tres secuencias):
  V_a,mk = 1 − (2·Z1_mk + Z0_mk) / (2·Z1_kk + Z0_kk)
  ← tensión de la FASE FALLADA, no de secuencia positiva. Es la misma
    magnitud con que se determinó el criterio de susceptibilidad del
    inversor por simulación dinámica (§3.4.2 de la memoria).

FALLA TRIFÁSICA EN PUNTO p DE LÍNEA k–j — Ecs. 3.16, 3.8:
  Z1_mp = (1−ξ)·Z1_mk + ξ·Z1_mj
  Z1_pp = (1−ξ)²·Z1_kk + ξ²·Z1_jj + 2ξ(1−ξ)·Z1_kj + ξ(1−ξ)·z1_kj
  V_mp  = 1 − Z1_mp / Z1_pp
  con ξ = Lkp / Lkj ∈ [0,1]

FALLA MONOFÁSICA EN PUNTO p DE LÍNEA k–j — Ecs. 3.16, 3.13:
  (mismas fórmulas geométricas para Z0_mp y Z0_pp)
  V_a,mp = 1 − (2·Z1_mp + Z0_mp) / (2·Z1_pp + Z0_pp)

NOTAS DE IMPLEMENTACIÓN
-----------------------
  - Filtro de cobertura en tension_falla_linea_1f: el cálculo monofásico
    sobre un corredor exige Z0_mk simulado en AMBOS extremos. Interpolar
    entre un extremo calibrado y otro algorítmico mezcla dos escalas
    distintas y producía perfiles imposibles (mínimos interiores cercanos
    a 0 p.u. en corredores cuyos dos extremos superaban 0,94 p.u.).
  - El clamp a [0,1] deja de ser silencioso: emite aviso por consola.
  - Rótulos y docstrings alineados con el criterio real del código, que es
    de desigualdad ESTRICTA (V < v_umbral).
"""

import numpy as np
from sic_datos import V_UMBRAL, V_NO_CRITICO


# ===========================================================================
# FALLAS EN BARRAS
# ===========================================================================

def tension_falla_barra_3f(Z1: np.ndarray, obs_buses: dict,
                           v_sim: dict = None) -> dict:
    """
    Tensión en barras de observación m ante falla trifásica en barra k.

    Si v_sim contiene el par (m, k) → usa el valor simulado directamente.
    En caso contrario → calcula V_mk = 1 - Z_mk/Z_kk (del algoritmo).

    Retorna dict {m: {k: |V_mk|}}
    """
    nbus = Z1.shape[0]
    v_sim = v_sim or {}
    res = {}
    for m in obs_buses:
        m_i = m - 1
        res[m] = {}
        for k in range(1, nbus + 1):
            if (m, k) in v_sim:
                res[m][k] = v_sim[(m, k)]          # exacto de simulación
            else:
                k_i = k - 1
                Zmk = Z1[m_i, k_i]
                Zkk = Z1[k_i, k_i]
                res[m][k] = abs(1.0 - Zmk / Zkk)   # calculado de Z_barra
    return res


def tension_falla_barra_1f(Z1: np.ndarray, Z0: np.ndarray,
                            obs_buses: dict,
                            z_mk_1: dict = None,
                            z_mk_0: dict = None,
                            v_sim_1f: dict = None,
                            v_sim_3f: dict = None) -> dict:
    """
    Tensión de fase real en barras de observación m ante falla 1φ en k.

    Fórmula exacta de tres secuencias:
      V_a_mk = 1 - (2·Z1_mk + Z0_mk) / (2·Z1_kk + Z0_kk)

    Jerarquía de fuentes para Z1_mk y Z0_mk:
      1. z_mk_1[(m,k)] y z_mk_0[(m,k)] — datos simulados directos
         (ZMK_1 y Z_MK_0 de sic_datos). Caso preferido: Z1_mk igual al
         de la rama 3φ (no se inventa un segundo valor), Z0_mk real.
      2. Si falta z_mk_0 pero hay v_sim_3f[(m,k)]: recalibra Z1_mk desde
         ese dato y usa Z0[m,k] del algoritmo como respaldo.
      3. Fallback puro: Z1[m,k] y Z0[m,k] del algoritmo de red.

    Retorna dict {m: {k: |V_a_mk|}}
    """
    nbus     = Z1.shape[0]
    z_mk_1   = z_mk_1   or {}
    z_mk_0   = z_mk_0   or {}
    v_sim_1f = v_sim_1f or {}
    v_sim_3f = v_sim_3f or {}

    res = {}
    for m in obs_buses:
        m_i = m - 1
        res[m] = {}
        for k in range(1, nbus + 1):
            k_i  = k - 1
            Z1kk = Z1[k_i, k_i]
            Z0kk = Z0[k_i, k_i]

            if (m, k) in z_mk_1 and (m, k) in z_mk_0:
                # Caso 1: formula exacta con datos simulados
                Z1mk = 1j * z_mk_1[(m, k)]
                Z0mk = 1j * z_mk_0[(m, k)]
                res[m][k] = abs(1.0 - (2*Z1mk + Z0mk) / (2*Z1kk + Z0kk))
            elif (m, k) in v_sim_1f:
                # Caso 2a: dato de tension directa (respaldo)
                res[m][k] = v_sim_1f[(m, k)]
            else:
                # Caso 2b/3: Z1_mk desde V_sim_3f o algoritmo; Z0_mk algoritmo
                if (m, k) in v_sim_3f:
                    Z1mk = 1j * (1.0 - v_sim_3f[(m, k)]) * Z1kk.imag
                else:
                    Z1mk = Z1[m_i, k_i]
                Z0mk = Z0[m_i, k_i]
                Ifk1 = 1.0 / (2*Z1kk + Z0kk)
                res[m][k] = abs(1.0 - Z1mk * Ifk1)
    return res


# ===========================================================================
# FALLAS EN TRAMOS DE LÍNEA
# ===========================================================================

def tension_falla_linea_3f(
    Z1: np.ndarray,
    corredores: list,
    obs_buses: dict,
    xi_vals: np.ndarray = None,
    v_sim: dict = None,
    zkj_sim: dict = None,
    v_umbral: float = V_UMBRAL,
) -> dict:
    """
    Tensión en barras de observación m ante falla 3φ en punto p de línea k–j.

    Numerador Z_mp = (1-ξ)Z_mk + ξZ_mj  → exacto desde V_sim (calibrado)
    Denominador Z_pp usa Z_kj exacto desde zkj_sim si disponible,
    algoritmo en caso contrario.

    Lógica de cortocircuito:
      - Si ambos extremos V_mk_eff > v_umbral → tramo no crítico (V=V_NO_CRITICO)
      - Si al menos un extremo es crítico → calcula con Ec. 3.16
    """
    if xi_vals is None:
        xi_vals = np.linspace(0.0, 1.0, 200)  # curva continua
    v_sim  = v_sim  or {}
    zkj_sim = zkj_sim or {}

    res = {}
    for m in obs_buses:
        m_i = m - 1
        res[m] = {}
        for c in corredores:
            k, j   = c['from'], c['to']
            k_i, j_i = k - 1, j - 1
            label  = c['label']

            # Z_kj: exacto desde simulación si disponible, algoritmo si no
            # Z_kj: impedancia de transferencia desde simulación (Ec. 3.41)
            if (k, j) in zkj_sim:
                Z1kj_sim = 1j * zkj_sim[(k, j)]
            else:
                Z1kj_sim = Z1[k_i, j_i]   # fallback: elemento off-diagonal Z_barra

            # z_kj: impedancia SERIE de un circuito individual (Ec. 3.16)
            z1_kj_serie = 1j * c['x1_pu_circ']

            Z1kk = Z1[k_i, k_i]
            Z1jj = Z1[j_i, j_i]

            # Tensiones efectivas en extremos
            v_mk = v_sim.get((m, k), V_NO_CRITICO)
            v_mj = v_sim.get((m, j), V_NO_CRITICO)

            if v_mk >= v_umbral and v_mj >= v_umbral:
                res[m][label] = [(xi, float("inf")) for xi in xi_vals]
                continue

            pts = []
            for xi in xi_vals:
                Z1mk_xi = Z1[m_i, k_i]
                Z1mj_xi = Z1[m_i, j_i]
                Z1mp = (1 - xi)*Z1mk_xi + xi*Z1mj_xi
                Z1pp = ((1-xi)**2*Z1kk + xi**2*Z1jj
                        + 2*xi*(1-xi)*Z1kj_sim       # transferencia desde simulación
                        + xi*(1-xi)*z1_kj_serie)     # serie circuito individual
                Vmp  = 1.0 - Z1mp / Z1pp
                pts.append((xi, abs(Vmp)))
            res[m][label] = pts
    return res


def tension_falla_linea_1f(
    Z1: np.ndarray,
    Z0: np.ndarray,
    corredores: list,
    obs_buses: dict,
    xi_vals: np.ndarray = None,
    v_sim_3f: dict = None,
    v_sim_1f: dict = None,
    zkj_sim: dict = None,
    v_umbral: float = V_UMBRAL,
    z_mk_1: dict = None,
    z_mk_0: dict = None,
) -> dict:
    """
    Tensión de fase real en barra de observación m ante falla 1φ en línea k–j.

    Fórmula exacta de tres secuencias:
      Z1_mp = (1-ξ)·Z1_mk + ξ·Z1_mj          (numerador sec. positiva)
      Z0_mp = (1-ξ)·Z0_mk + ξ·Z0_mj          (numerador sec. cero)
      Z1_pp, Z0_pp = interpolación cuadrática habitual
      I_fp1 = 1 / (2·Z1_pp + Z0_pp)
      V_a_mp = 1 - (2·Z1_mp + Z0_mp) · I_fp1

    z_mk_1 y z_mk_0: impedancias de transferencia simuladas
      (Z_MK_1, Z_MK_0 de sic_datos). Cuando están disponibles para los
      extremos k y j del corredor, reemplazan los elementos off-diagonal
      de Z_barra (que son del algoritmo), igual que calibrar_zbarra_sim
      ya hace con Z1_mk para la rama 3φ. Eliminan el parche Z1_eff_1f
      previo, que era metodológicamente inconsistente (creaba un segundo
      valor de Z1_mk distinto al de la rama 3φ).

    Si faltan z_mk_0 para algún extremo, usa Z0[m,k] del algoritmo.
    Si faltan z_mk_1 para algún extremo, usa Z1[m,k] del algoritmo.
    """
    if xi_vals is None:
        xi_vals = np.linspace(0.0, 1.0, 200)
    v_sim_3f = v_sim_3f or {}
    v_sim_1f = v_sim_1f or {}
    zkj_sim  = zkj_sim  or {}
    z_mk_1   = z_mk_1   or {}
    z_mk_0   = z_mk_0   or {}

    def _v_eff_1f(m, k):
        """Tensión efectiva en extremo de corredor — para filtro de criticidad."""
        if (m, k) in v_sim_1f: return v_sim_1f[(m, k)]
        if (m, k) in v_sim_3f: return v_sim_3f[(m, k)]
        return V_NO_CRITICO

    res = {}
    for m in obs_buses:
        m_i = m - 1
        res[m] = {}
        for c in corredores:
            k, j   = c['from'], c['to']
            k_i, j_i = k - 1, j - 1
            label  = c['label']

            # Z_kj transferencia desde simulación (Ec. 3.41)
            if (k, j) in zkj_sim:
                Z1kj_sim = 1j * zkj_sim[(k, j)]
                Z0kj_sim = 1j * zkj_sim[(k, j)]
            else:
                Z1kj_sim = Z1[k_i, j_i]
                Z0kj_sim = Z0[k_i, j_i]

            z1_kj_serie = 1j * c['x1_pu_circ']
            z0_kj_serie = 1j * c['x0_pu_circ']

            Z1kk = Z1[k_i, k_i];  Z1jj = Z1[j_i, j_i]
            Z0kk = Z0[k_i, k_i];  Z0jj = Z0[j_i, j_i]

            # ── Filtro de cobertura ─────────────────────────────────
            # La interpolación de Z0_mp exige dato simulado en AMBOS
            # extremos. Si falta en uno, la interpolación mezclaría un
            # valor calibrado con uno algorítmico de escala distinta.
            if (m, k) not in z_mk_0 or (m, j) not in z_mk_0:
                res[m][label] = [(xi, float("inf")) for xi in xi_vals]
                continue

            # Filtro de criticidad
            v_mk = _v_eff_1f(m, k)
            v_mj = _v_eff_1f(m, j)
            if v_mk >= v_umbral and v_mj >= v_umbral:
                res[m][label] = [(xi, float("inf")) for xi in xi_vals]
                continue

            # Z1_mk/Z1_mj: dato simulado si disponible, algoritmo si no
            Z1mk = 1j * z_mk_1[(m, k)] if (m, k) in z_mk_1 else Z1[m_i, k_i]
            Z1mj = 1j * z_mk_1[(m, j)] if (m, j) in z_mk_1 else Z1[m_i, j_i]

            # Z0_mk/Z0_mj: dato simulado si disponible, algoritmo si no
            Z0mk = 1j * z_mk_0[(m, k)] if (m, k) in z_mk_0 else Z0[m_i, k_i]
            Z0mj = 1j * z_mk_0[(m, j)] if (m, j) in z_mk_0 else Z0[m_i, j_i]

            pts = []
            v_max_bruto = 0.0
            for xi in xi_vals:
                Z1mp = (1-xi)*Z1mk + xi*Z1mj
                Z1pp = ((1-xi)**2*Z1kk + xi**2*Z1jj
                        + 2*xi*(1-xi)*Z1kj_sim
                        + xi*(1-xi)*z1_kj_serie)
                Z0mp = (1-xi)*Z0mk + xi*Z0mj
                Z0pp = ((1-xi)**2*Z0kk + xi**2*Z0jj
                        + 2*xi*(1-xi)*Z0kj_sim
                        + xi*(1-xi)*z0_kj_serie)
                denom = 2*Z1pp + Z0pp
                if abs(denom) < 1e-8:          # guarda numerica: polo espurio
                    pts.append((xi, V_NO_CRITICO))
                    continue
                Ifp1  = 1.0 / denom
                V_amp = 1.0 - (2*Z1mp + Z0mp) * Ifp1
                v_amp = abs(V_amp)
                v_max_bruto = max(v_max_bruto, v_amp)
                pts.append((xi, min(v_amp, 1.0)))      # clamp fisico [0,1]
            if v_max_bruto > 1.0 + 1e-6:               # no enmascarar
                print(f"  [aviso] V_a > 1 p.u. en {label} (m={m}): "
                      f"maximo {v_max_bruto:.4f} p.u.")
            res[m][label] = pts
    return res


# ===========================================================================
# IDENTIFICACIÓN DE ÁREAS DE VULNERABILIDAD
# ===========================================================================

def areas_vulnerabilidad_barras(
    v3f: dict, v1f: dict, obs_buses: dict,
    nombres: dict, v_umbral: float = V_UMBRAL,
) -> dict:
    """
    Identifica barras cuya falla produce V_mk < v_umbral (desigualdad
    estricta) en la barra de observación m.

    Retorna
    -------
    dict {m: {'3f': [lista_barras_criticas], '1f': [lista_barras_criticas]}}
    """
    areas = {}
    for m, m_nombre in obs_buses.items():
        criticas_3f = [k for k, v in v3f[m].items() if v < v_umbral]
        criticas_1f = [k for k, v in v1f[m].items() if v < v_umbral]
        areas[m] = {'3f': criticas_3f, '1f': criticas_1f}
    return areas


def areas_vulnerabilidad_lineas(
    vl3f: dict, vl1f: dict, obs_buses: dict, v_umbral: float = V_UMBRAL,
) -> dict:
    """
    Identifica corredores cuya falla (en algún ξ) produce V_mp < v_umbral
    (desigualdad estricta).

    Retorna
    -------
    dict {m: {'3f': [label_criticos], '1f': [label_criticos]}}
    """
    areas = {}
    for m in obs_buses:
        criticas_3f = [lbl for lbl, pts in vl3f[m].items()
                       if any(v < v_umbral for _, v in pts)]
        criticas_1f = [lbl for lbl, pts in vl1f[m].items()
                       if any(v < v_umbral for _, v in pts)]
        areas[m] = {'3f': criticas_3f, '1f': criticas_1f}
    return areas


def curvas_area_vulnerabilidad(
    vl3f: dict, vl1f: dict,
    areas_l: dict, corredores: list,
    obs_buses: dict,
) -> dict:
    """
    Para cada barra de observación m, retorna las curvas V(ξ) completas
    solo para los corredores críticos de su área de vulnerabilidad.

    Los puntos se entregan como (ξ, V), con ξ ∈ [0,1]; longitud_km se
    adjunta por separado en la clave 'L_km'.

    Retorna
    -------
    dict {m: [{'label', 'L_km', 'lambda_anual',
               'critico_3f', 'critico_1f',
               'pts_3f': [(km, V), ...],
               'pts_1f': [(km, V), ...]}]}
    """
    corr_dict = {c['label']: c for c in corredores}
    result = {}
    for m in obs_buses:
        criticos_3f = set(areas_l[m]['3f'])
        criticos_1f = set(areas_l[m]['1f'])
        criticos = criticos_3f | criticos_1f
        curvas = []
        for lbl in criticos:
            c = corr_dict[lbl]
            L = c['longitud_km']
            pts3 = list(vl3f[m][lbl])   # ya son (xi, V)
            pts1 = list(vl1f[m][lbl])
            curvas.append({
                'label':       lbl,
                'L_km':        L,
                'lambda_anual': 0.7 * L / 100.0,
                'critico_3f':  lbl in criticos_3f,
                'critico_1f':  lbl in criticos_1f,
                'pts_3f':      pts3,
                'pts_1f':      pts1,
            })
        result[m] = curvas
    return result




def imprimir_tensiones_barras(
    v3f: dict, v1f: dict,
    obs_buses: dict, nombres: dict, v_umbral: float = V_UMBRAL,
) -> None:
    """Tabla de tensiones de falla por barra."""
    nbus = len(v3f[list(obs_buses.keys())[0]])
    for m, m_nombre in obs_buses.items():
        print(f"\n{'='*70}")
        print(f"  Barra de observación: {m_nombre}  (barra {m})")
        print(f"  Umbral de conmutación: V < {v_umbral} p.u.")
        print(f"{'='*70}")
        print(f"  {'k':>3}  {'Barra de falla':<24}  "
              f"{'|V_mk| 3φ':>10}  {'|V_a,mk| 1φ':>13}  "
              f"{'3φ':>4}  {'1φ':>4}")
        print(f"  {'-'*3}  {'-'*24}  {'-'*10}  {'-'*13}  {'-'*4}  {'-'*4}")
        for k in range(1, nbus + 1):
            v3 = v3f[m][k]
            v1 = v1f[m][k]
            flag3 = '⚠' if v3 < v_umbral else ''
            flag1 = '⚠' if v1 < v_umbral else ''
            s3 = f'>{v_umbral}' if not np.isfinite(v3) or v3 >= 0.8999 else f'{v3:>10.4f}'
            s1 = f'>{v_umbral}' if not np.isfinite(v1) or v1 >= 0.8999 else f'{v1:>10.4f}'
            print(f"  {k:>3}  {nombres.get(k,'?'):<24}  "
                  f"{s3:>10}  {s1:>13}  {flag3:>4}  {flag1:>4}")


def imprimir_tensiones_lineas(
    vl3f: dict, vl1f: dict,
    obs_buses: dict, v_umbral: float = V_UMBRAL,
) -> None:
    """Resumen de tensiones mínimas por corredor de línea."""
    for m, m_nombre in obs_buses.items():
        print(f"\n{'='*80}")
        print(f"  Barra de observación: {m_nombre}  — Fallas en tramos de línea")
        print(f"{'='*80}")
        print(f"  {'Corredor':<40}  "
              f"{'Vmin 3φ':>8}  {'ξ':>5}  "
              f"{'Vmin 1φ':>8}  {'ξ':>5}  {'Crítico':>7}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*7}")
        for lbl in vl3f[m]:
            pts3 = vl3f[m][lbl]
            pts1 = vl1f[m][lbl]
            vmin3, xi3 = min(pts3, key=lambda p: p[1])
            vmin1, xi1 = min(pts1, key=lambda p: p[1])
            xi3_val   = min(pts3, key=lambda p: p[1])[0]
            vmin3_val = min(pts3, key=lambda p: p[1])[1]
            xi1_val   = min(pts1, key=lambda p: p[1])[0]
            vmin1_val = min(pts1, key=lambda p: p[1])[1]
            critico = '⚠' if vmin3_val < v_umbral or vmin1_val < v_umbral else ''
            # Formatear: inf → ">umbral"
            s3 = f'>{v_umbral}' if not np.isfinite(vmin3_val) else f'{vmin3_val:>8.4f}'
            s1 = f'>{v_umbral}' if not np.isfinite(vmin1_val) else f'{vmin1_val:>8.4f}'
            x3 = f'{xi3_val:>5.2f}' if np.isfinite(vmin3_val) else '  ---'
            x1 = f'{xi1_val:>5.2f}' if np.isfinite(vmin1_val) else '  ---'
            print(f"  {lbl:<40}  {s3:>8}  {x3}  {s1:>8}  {x1}  {critico:>7}")


def imprimir_areas_vulnerabilidad(
    areas_b: dict, areas_l: dict,
    obs_buses: dict, nombres: dict,
) -> None:
    """
    Resumen de áreas de vulnerabilidad — formato tabular unificado.

    En vez de listar las barras/líneas críticas dos veces (una para 3φ,
    otra para 1φ, con la lista 1φ casi siempre subconjunto de la 3φ), se
    presenta una sola tabla por barra de observación, con columnas que
    indican si cada elemento es crítico bajo cada tipo de falla.
    """
    ANCHO = 72

    for m, m_nombre in obs_buses.items():
        crit_b3 = set(areas_b[m]['3f'])
        crit_b1 = set(areas_b[m]['1f'])
        crit_l3 = list(areas_l[m]['3f'])
        crit_l1 = set(areas_l[m]['1f'])

        # Unión preservando orden de aparición (3φ primero, luego 1φ-only)
        todas_b = sorted(crit_b3 | crit_b1)
        todas_l = list(crit_l3) + [lbl for lbl in crit_l1 if lbl not in crit_l3]

        print(f"\n{'='*ANCHO}")
        print(f"  {m_nombre}  (barra {m})")
        print('='*ANCHO)
        print(f"  Barras críticas:  3φ = {len(crit_b3):2d}   1φ = {len(crit_b1):2d}")
        print(f"  Líneas críticas:  3φ = {len(crit_l3):2d}   1φ = {len(crit_l1):2d}")

        print(f"\n  {'Barra':<26}{'3φ':>6}{'1φ':>6}")
        print(f"  {'-'*26}{'-'*6}{'-'*6}")
        if todas_b:
            for k in todas_b:
                c3 = '   X' if k in crit_b3 else '    '
                c1 = '   X' if k in crit_b1 else '    '
                print(f"  {k:>3d}  {nombres.get(k,'?'):<21}{c3:>6}{c1:>6}")
        else:
            print("  (ninguna)")

        print(f"\n  {'Corredor':<42}{'3φ':>6}{'1φ':>6}")
        print(f"  {'-'*42}{'-'*6}{'-'*6}")
        if todas_l:
            for lbl in todas_l:
                c3 = '   X' if lbl in crit_l3 else '    '
                c1 = '   X' if lbl in crit_l1 else '    '
                print(f"  {lbl:<42}{c3:>6}{c1:>6}")
        else:
            print("  (ninguna)")
    print(f"\n{'='*ANCHO}")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sic_datos   import DATA1, DATA0, NBUS, NOMBRES, BARRAS_OBS, V_UMBRAL, get_corredores
    from zbarra      import construir_zbarra

    Z1 = construir_zbarra(DATA1, NBUS)
    Z0 = construir_zbarra(DATA0, NBUS)
    corredores = get_corredores()
    xi = np.linspace(0.0, 1.0, 21)

    v3f  = tension_falla_barra_3f(Z1, BARRAS_OBS)
    v1f  = tension_falla_barra_1f(Z1, Z0, BARRAS_OBS)
    vl3f = tension_falla_linea_3f(Z1, corredores, BARRAS_OBS, xi)
    vl1f = tension_falla_linea_1f(Z1, Z0, corredores, BARRAS_OBS, xi)

    imprimir_tensiones_barras(v3f, v1f, BARRAS_OBS, NOMBRES, V_UMBRAL)
    imprimir_tensiones_lineas(vl3f, vl1f, BARRAS_OBS, V_UMBRAL)

    areas_b = areas_vulnerabilidad_barras(v3f, v1f, BARRAS_OBS, NOMBRES, V_UMBRAL)
    areas_l = areas_vulnerabilidad_lineas(vl3f, vl1f, BARRAS_OBS, V_UMBRAL)
    imprimir_areas_vulnerabilidad(areas_b, areas_l, BARRAS_OBS, NOMBRES)
