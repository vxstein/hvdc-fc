"""
zbarra.py — Construcción de la Matriz de Impedancias de Barra
=============================================================
Implementa el algoritmo del Apéndice B de la memoria (VR, 2026),
basado en incorporación secuencial de elementos con Reducción de Kron.

Referencia de ecuaciones:
  Paso 1 (id=0): Ec. B.1  — rama al nodo de referencia
  Paso 2 (id=1): Ec. B.2  — nueva barra conectada a barra existente
  Paso 3 (id=-1): Ecs. B.3–B.5 — enlace entre barras existentes + Kron
"""

import numpy as np


def construir_zbarra(data: np.ndarray, nbus: int) -> np.ndarray:
    """
    Construye la matriz de impedancias de barra Z (nbus × nbus, compleja).

    Parámetros
    ----------
    data : ndarray shape (n, 4)
        Columnas: [barra_origen, barra_destino, X_pu, id]
        id =  0 → rama al nodo de referencia
        id =  1 → rama radial (agrega nueva barra)
        id = -1 → enlace de malla (Reducción de Kron)
    nbus : int
        Número total de barras del sistema.

    Retorna
    -------
    Z : ndarray shape (nbus, nbus), dtype complex
        Z[i, j] = impedancia entre barras (i+1) y (j+1).
        Los elementos diagonales Z[k,k] son las impedancias de Thévenin.
        Los elementos fuera de la diagonal Z[m,k] son las impedancias de
        transferencia usadas para calcular tensiones de falla.
    """
    fb  = data[:, 0].astype(int)
    tb  = data[:, 1].astype(int)
    x   = data[:, 2]              # reactancia en p.u. (real)
    ids = data[:, 3].astype(int)

    # Z_work: matriz que crece / se modifica durante el algoritmo
    Z_work = np.zeros((0, 0), dtype=complex)
    # Mapa barra_numero → índice en Z_work
    bus_idx: dict[int, int] = {}

    for k in range(len(fb)):
        xk  = 1j * x[k]       # impedancia pura imaginaria
        p   = fb[k]
        q   = tb[k]

        # ------------------------------------------------------------------
        # PASO 1: Rama al nodo de referencia (id=0) — Ec. B.1
        # ------------------------------------------------------------------
        if ids[k] == 0:
            idx_q = len(bus_idx)
            bus_idx[q] = idx_q
            n = len(Z_work)
            Z_new = np.zeros((n + 1, n + 1), dtype=complex)
            Z_new[:n, :n] = Z_work
            Z_new[n, n]   = xk
            Z_work = Z_new

        # ------------------------------------------------------------------
        # PASO 2: Nueva barra conectada a barra existente (id=1) — Ec. B.2
        # ------------------------------------------------------------------
        elif ids[k] == 1:
            if p not in bus_idx:
                raise ValueError(
                    f"Barra {p} no existe aún al procesar rama {p}→{q} (id=1). "
                    "Verificar orden del árbol en DATA."
                )
            p_i  = bus_idx[p]
            q_i  = len(bus_idx)
            bus_idx[q] = q_i
            n = len(Z_work)
            Z_new = np.zeros((n + 1, n + 1), dtype=complex)
            Z_new[:n, :n]  = Z_work
            Z_new[:n, n]   = Z_work[:, p_i]        # columna p copiada
            Z_new[n, :n]   = Z_work[p_i, :]        # fila p copiada (simetría)
            Z_new[n, n]    = Z_work[p_i, p_i] + xk # Zpp + zpq
            Z_work = Z_new

        # ------------------------------------------------------------------
        # PASO 3: Enlace de malla (id=-1) — Ecs. B.3–B.5 + Ec. B.4 (Kron)
        # ------------------------------------------------------------------
        elif ids[k] == -1:
            q_i = bus_idx[q]

            if p == 0:
                # Enlace entre barra q y el nodo de referencia — Ec. B.5
                Zll = xk + Z_work[q_i, q_i]
                dZ  = -Z_work[:, q_i].reshape(-1, 1)
            else:
                # Enlace entre barras p y q existentes — Ec. B.3
                p_i = bus_idx[p]
                Zll = (xk + Z_work[q_i, q_i] + Z_work[p_i, p_i]
                       - 2 * Z_work[p_i, q_i])
                dZ  = (Z_work[:, q_i] - Z_work[:, p_i]).reshape(-1, 1)

            # Reducción de Kron — Ec. B.4
            Z_work = Z_work - (dZ @ dZ.conj().T) / Zll

    # ------------------------------------------------------------------
    # Reordenar para que Z[i,j] corresponda a barras (i+1, j+1)
    # ------------------------------------------------------------------
    Z_out = np.zeros((nbus, nbus), dtype=complex)
    for bnum, idx in bus_idx.items():
        if 1 <= bnum <= nbus:
            for bnum2, idx2 in bus_idx.items():
                if 1 <= bnum2 <= nbus:
                    Z_out[bnum - 1, bnum2 - 1] = Z_work[idx, idx2]

    return Z_out


def imprimir_zbarra(Z: np.ndarray, nombres: dict, titulo: str = "Z_barra") -> None:
    """Imprime los elementos diagonales y algunas transferencias clave."""
    nbus = Z.shape[0]
    print(f"\n{'='*60}")
    print(f"  {titulo}  —  elementos diagonales (impedancias de Thévenin)")
    print(f"{'='*60}")
    print(f"  {'Barra':>5}  {'Nombre':<25}  {'|Zkk| [p.u.]':>13}  {'Xkk [p.u.]':>12}")
    print(f"  {'-'*5}  {'-'*25}  {'-'*13}  {'-'*12}")
    for k in range(nbus):
        Zkk = Z[k, k]
        print(f"  {k+1:>5}  {nombres.get(k+1,'?'):<25}  "
              f"{abs(Zkk):>13.5f}  {Zkk.imag:>12.5f}")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sic_datos import DATA1, DATA0, NBUS, NOMBRES

    print("Construyendo Z_barra secuencia positiva …")
    Z1 = construir_zbarra(DATA1, NBUS)
    imprimir_zbarra(Z1, NOMBRES, "Z1_barra (secuencia positiva)")

    print("\nConstruyendo Z_barra secuencia cero …")
    Z0 = construir_zbarra(DATA0, NBUS)
    imprimir_zbarra(Z0, NOMBRES, "Z0_barra (secuencia cero)")


def aplicar_zkk_simulacion(Z: np.ndarray, Z1kk: list, Z0kk: list) -> tuple:
    """
    Reemplaza los elementos diagonales de Z1_barra y Z0_barra con los
    valores obtenidos de la simulación (Cuadro 5.1).

    Los elementos fuera de la diagonal (impedancias de transferencia Z_mk)
    se mantienen del algoritmo de red, que es la mejor estimación disponible.

    Parámetros
    ----------
    Z    : Z_barra (25×25) del algoritmo — solo se usa para off-diagonal
    Z1kk : lista de 25 valores Z1_kk desde Scc3 simulada
    Z0kk : lista de 25 valores Z0_kk desde Scc1 simulada / ratio red

    Retorna
    -------
    Z1, Z0 : matrices (25×25) con diagonal corregida
    """
    nbus = Z.shape[0]
    Z1 = Z.copy()
    Z0 = Z.copy()

    # Escalar off-diagonal proporcionalmente al cambio en diagonal
    # Z_mk_correg = Z_mk_algo × sqrt(Z1kk_sim[m] / Z1kk_algo[m])
    #                         × sqrt(Z1kk_sim[k] / Z1kk_algo[k])
    # (aproximación geométrica que preserva la simetría)
    Z1kk_algo = np.array([Z[k, k].imag for k in range(nbus)])
    Z0kk_algo = np.array([Z[k, k].imag for k in range(nbus)])  # misma base

    for m in range(nbus):
        f_m1 = np.sqrt(Z1kk[m] / Z1kk_algo[m]) if Z1kk_algo[m] > 0 else 1.0
        for k in range(nbus):
            if m == k:
                Z1[m, k] = 1j * Z1kk[m]
                Z0[m, k] = 1j * Z0kk[m]
            else:
                f_k1 = np.sqrt(Z1kk[k] / Z1kk_algo[k]) if Z1kk_algo[k] > 0 else 1.0
                Z1[m, k] = 1j * Z.imag[m, k] * f_m1 * f_k1
                Z0[m, k] = 1j * Z.imag[m, k] * f_m1 * f_k1  # misma escala approx

    return Z1, Z0


def calibrar_zbarra_sim(
    Z_algo: np.ndarray,
    Z1kk_sim: list,
    Z0kk_sim: list,
    V_sim_3f: dict,
    obs_buses: list = None,
    v_default: float = 0.9,
    Z_algo_0: np.ndarray = None,
) -> tuple:
    """
    Construye Z1_barra y Z0_barra calibradas con datos de simulación.

    Diagonal (exacta):
      Z1[k,k] = j × Z1kk_sim[k]    (de Scc3)
      Z0[k,k] = j × Z0kk_sim[k]    (de Scc1)

    Off-diagonal Z1_mk para filas de barras de observación — cada par NO
    ORDENADO {m,k} se calcula una sola vez (no depende del orden de
    iteración de obs_buses), con la siguiente precedencia:

      (a) Si (m,k) o (k,m) está en V_sim_3f → Z1_mk = (1-V_sim) × Z1kk_sim[lado con dato]
          EXACTO, desde simulación.
      (b) Si no hay dato Y k también es barra de observación → caso ambiguo:
          no existe un "lado k" único, porque ambas direcciones son válidas
          (V_mk con falla en k, V_km con falla en m). Se usa
          Z1_mk = (1-v_default) × min(Z1kk_sim[m], Z1kk_sim[k]), lo que
          garantiza V_mk ≥ v_default Y V_km ≥ v_default simultáneamente,
          sin depender de cuál de las dos barras se procesó primero.
      (c) Si no hay dato y k NO es barra de observación → caso simple,
          sin ambigüedad: Z1_mk = (1-v_default) × Z1kk_sim[k].

    Off-diagonal entre barras no-observación: del algoritmo (aceptable,
    solo afecta Z_pp en el denominador, no los valores de V en obs-buses).

    Off-diagonal Z0_mk: del algoritmo de secuencia cero (Z_algo_0, construido
    desde DATA0). Los pares con dato simulado se resuelven aguas abajo, en
    tensiones_falla.py, mediante Z_MK_0.

    ADVERTENCIA DE ESCALA. La matriz algorítmica no incluye las
    fuentes de generación, por lo que su diagonal es entre 1 y 24 veces
    mayor que la obtenida de Scc. Al sustituir la diagonal por los valores
    simulados y conservar los elementos fuera de ella, la matriz resultante
    deja de cumplir |Z_mk| <= min(Z_mm, Z_kk). Por eso NO debe usarse un
    elemento algorítmico fuera de diagonal junto a uno calibrado en una
    misma expresión: es el origen del defecto corregido mediante
    el filtro de cobertura de tension_falla_linea_1f.

    Parámetros
    ----------
    Z_algo    : Z_barra del algoritmo, secuencia positiva (DATA1, 25×25)
    Z_algo_0  : Z_barra del algoritmo, secuencia cero (DATA0, 25×25).
                Si es None se reutiliza Z_algo, comportamiento incorrecto
                que dejaba la matriz homopolar como copia de la de
                secuencia positiva.
    Z1kk_sim  : lista de 25 Z1_kk desde Scc3 simulada
    Z0kk_sim  : lista de 25 Z0_kk desde Scc1 simulada
    V_sim_3f  : dict {(m, k): V_mk} — tensiones 3φ simuladas
    obs_buses : barras de observación (default: [4, 15, 24])
    v_default : tensión límite para pares no calibrados (default 0.9,
                ver sic_datos.V_NO_CRITICO)

    Retorna
    -------
    Z1_cal, Z0_cal : matrices (25×25) calibradas
    """
    nbus = Z_algo.shape[0]
    Z1 = Z_algo.copy()
    Z0 = (Z_algo_0 if Z_algo_0 is not None else Z_algo).copy()
    obs_buses = obs_buses or [4, 15, 24]
    obs_set = set(obs_buses)

    # ── 1. Diagonal exacta desde simulación ──────────────────────────────
    for k in range(nbus):
        Z1[k, k] = 1j * Z1kk_sim[k]
        Z0[k, k] = 1j * Z0kk_sim[k]

    # ── 2. Filas de barras de observación: consistentes con Z_kk_sim ─────
    # Problema sin este paso: Z_mk del algoritmo es inconsistente con
    # Z_kk_sim → ratio Z_mk/Z_kk erróneo → tensiones de línea incorrectas.
    #
    # Cada par no-ordenado se calcula una sola vez (set `procesados`) para
    # eliminar la dependencia del orden de iteración — ver docstring (a)-(c).
    procesados = set()
    for m in obs_buses:
        m_i = m - 1
        for k_i in range(nbus):
            if m_i == k_i:
                continue
            k = k_i + 1
            par = frozenset((m, k))
            if par in procesados:
                continue  # ya calculado (y simétrico) en una iteración previa

            if (m, k) in V_sim_3f:
                z_mk = 1j * (1.0 - V_sim_3f[(m, k)]) * Z1kk_sim[k_i]
            elif (k, m) in V_sim_3f:
                z_mk = 1j * (1.0 - V_sim_3f[(k, m)]) * Z1kk_sim[m_i]
            elif k in obs_set:
                z_mk = 1j * (1.0 - v_default) * min(Z1kk_sim[m_i], Z1kk_sim[k_i])
            else:
                z_mk = 1j * (1.0 - v_default) * Z1kk_sim[k_i]

            Z1[m_i, k_i] = z_mk
            Z1[k_i, m_i] = z_mk              # simetría Z_mk = Z_km
            procesados.add(par)

    return Z1, Z0


def calibrar_zmk_1f_simulado(
    V_sim_1f: dict,
    Z1: np.ndarray,
    Z0: np.ndarray,
) -> dict:
    """
    Calibra una impedancia de transferencia efectiva Z1_mk, exclusiva
    para la rama de fallas monofásicas, directamente desde tensiones
    simuladas V_sim_1f.

    Se obtiene invirtiendo la misma fórmula usada en
    tension_falla_barra_1f para reconstruir V^(1)_mk desde Z1_mk:

        V^(1)_mk = 1 - Z1_mk · I^(0)_fk ,   I^(0)_fk = 1 / (2·Z1_kk + Z0_kk)
        ⟹ Z1_mk_eff_1F = (1 - V_sim_1f) · (2·Z1_kk + Z0_kk)

    Z1_kk y Z0_kk son los elementos diagonales YA calibrados desde
    Scc3φ/Scc1φ (independientes de V_sim_1f).

    Este Z1_mk_eff_1F se usa luego en tension_falla_linea_1f en vez del
    Z1_mk calibrado solo con datos trifásicos (V_sim_3f) — sin este paso,
    el barrido de línea para fallas monofásicas no reproduce los valores
    simulados en los extremos de cada corredor (solo coincide la rama 3φ,
    porque Z1 sí queda calibrado con V_sim_3f en calibrar_zbarra_sim).

    Parámetros
    ----------
    V_sim_1f : dict {(m, k): V_mk} — tensiones 1φ simuladas (fase real)
    Z1, Z0   : matrices (25×25) ya calibradas (diagonal exacta)

    Retorna
    -------
    dict {(m, k): Z1_mk_eff_1F} — solo para los pares con dato simulado
    """
    Z1_eff = {}
    for (m, k), v in V_sim_1f.items():
        k_i = k - 1
        Z1kk = Z1[k_i, k_i]
        Z0kk = Z0[k_i, k_i]
        Z1_eff[(m, k)] = (1.0 - v) * (2 * Z1kk + Z0kk)
    return Z1_eff
