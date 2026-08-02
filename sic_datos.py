"""
sic_datos.py — Parámetros del STN simplificado de 25 barras  (v1.0)
====================================================================
Fuente: VR (2026), Cuadros 4.1, 4.2 y 5.1

Bases: Sbase=100 MVA, Vbase_220=220 kV (Zbase=484 Ω), Vbase_500=500 kV (Zbase=2500 Ω)
"""
import numpy as np

SBASE     = 100.0
VBASE_220 = 220.0
VBASE_500 = 500.0
ZBASE_220 = VBASE_220**2 / SBASE   # 484 Ω
ZBASE_500 = VBASE_500**2 / SBASE   # 2500 Ω
NBUS      = 25
V_UMBRAL  = 0.9  # p.u. — umbral área de vulnerabilidad (V_mp < V_UMBRAL ⇒ crítico)

# Placeholder para pares (m,k) sin dato de simulación: límite inferior conocido
# que garantiza clasificación "no crítico" en calibrar_zbarra_sim y en las
# funciones de tensiones_falla.py. Con el criterio estrictamente "<" (no "≤"),
# coincidir exactamente con V_UMBRAL ya es seguro (0.9 < 0.9 es False), así
# que se reutiliza el mismo valor — no se necesita un margen artificial.
V_NO_CRITICO = V_UMBRAL  # p.u.

# Modelos de probabilidad de FC (Ecs. 4.1 y 4.2)
V_UMBRAL_3F   = 0.85   # umbral escalón falla trifásica (Ec. 4.1 / ec:pfc_trifasica)
V_TRANS_1F_HI = 0.9    # inicio transición lineal falla monofásica (P_FC=0) (Ec. 4.2 / ec:pfc_monofasica)
V_TRANS_1F_LO = 0.8    # fin transición (P_FC=1) falla monofásica (Ec. 4.2 / ec:pfc_monofasica)

NOMBRES = {
    1:'Paposo 220',        2:'D. de Almagro 220',  3:'Carrera Pinto 220',
    4:'Cardones 220',      5:'Maitencillo 220',     6:'Punta Colorada 220',
    7:'Pan de Azúcar 220', 8:'Los Vilos 220',       9:'Nogales 220',
    10:'Quillota 220',     11:'Polpaico 220',        12:'Polpaico 500',
    13:'Cerro Navia 220',  14:'Alto Jahuel 220',     15:'Alto Jahuel 500',
    16:'Ancoa 500',        17:'Ancoa 220',           18:'Itahue 220',
    19:'Charrúa 500',      20:'Charrúa 220',         21:'Concepción 220',
    22:'Temuco 220',       23:'Cautín 220',          24:'Valdivia 220',
    25:'Puerto Montt 220',
}

BARRAS_OBS = {4:'Cardones 220', 15:'Alto Jahuel 500', 24:'Valdivia 220'}

# ===========================================================================
# Cuadro 5.1: Potencias de cortocircuito de simulación DIgSILENT
# Listas indexadas 0..24 → barra 1..25
# ===========================================================================
SCC3_MVA = [
    367.10,  510.77,  721.62, 1292.37, 2280.97,
   1600.03, 1538.95, 2089.34, 4638.67, 8274.73,
   7949.67, 6871.32, 5711.92, 6111.86, 6970.92,
   7366.94, 4725.33, 2829.57, 7033.78, 7606.27,
   2339.57, 2287.57, 2319.55, 1464.26, 1080.99,
]

SCC1_MVA = [
    173.00,  218.21,  249.28,  563.32,  869.98,
    465.30,  587.35,  634.04, 1019.21, 3165.93,
   2269.14, 1992.09, 1829.71, 2531.19, 2444.07,
   2090.59, 1510.24, 1134.07, 2319.66, 3052.15,
   1068.92,  770.09,  770.03,  577.36,  427.46,
]


# ===========================================================================
# Tensiones de falla simuladas — DIgSILENT (Cuadro 5.2)
# Clave (m, k): tensión en barra de observación m ante falla en barra k.
# Solo se incluyen valores ≤ 0.90 p.u. (los que producen riesgo de falla
# de conmutación). Las entradas ausentes implican tensión > 0.90 p.u.
# ===========================================================================

V_SIM_3F = {
    # ── Cardones 220 (bus 4) ─────────────────────────────────────────────
    (4,  1): 0.787, (4,  2): 0.670, (4,  3): 0.488, (4,  4): 0.000,
    (4,  5): 0.003, (4,  6): 0.502, (4,  7): 0.639, (4,  8): 0.853,
    (4,  9): 0.895,
    # Nota: el par (4,15) NO tiene dato real de simulación, pero ya no
    # requiere un valor explícito aquí — calibrar_zbarra_sim (zbarra.py)
    # lo resuelve mediante la regla simétrica (mínimo de Z1kk_sim entre
    # ambas barras de observación), que garantiza V≥V_NO_CRITICO en ambas
    # direcciones sin depender del orden de iteración. Ver zbarra.py.
    # ── Alto Jahuel 500 (bus 15) ─────────────────────────────────────────
    (15, 10): 0.817, (15, 11): 0.630, (15, 12): 0.227,
    (15, 13): 0.832, (15, 14): 0.347, (15, 15): 0.000,
    (15, 16): 0.283, (15, 17): 0.729,
    (15, 19): 0.470, (15, 20): 0.638,
    # ── Valdivia 220 (bus 24) ────────────────────────────────────────────
    (24, 12): 0.873, (24, 15): 0.866, (24, 16): 0.758,
    (24, 19): 0.685, (24, 20): 0.476, (24, 22): 0.331,
    (24, 23): 0.315, (24, 24): 0.000, (24, 25): 0.570,
}

# ===========================================================================
# Impedancias de transferencia simuladas — DIgSILENT
# Clave (m, k): impedancia de transferencia Z_mk (en p.u.) entre barra
# de observación m y barra de falla k.
#
# Z_MK_1 — secuencia positiva (= secuencia negativa, red pasiva)
#   Fuente: mismos datos de V_SIM_3F, pero expresados directamente como
#   impedancia: Z1_mk = (1 - V_sim_3f) * Z1_kk.
#   Usados en tension_falla_barra_1f y tension_falla_linea_1f como
#   numerador Z1_mp = (1-xi)*Z1_mk + xi*Z1_mj (igual que en rama 3f).
#
# Z_MK_0 — secuencia cero
#   Fuente: simulación directa en PowerFactory (informe de secuencias).
#   Algunos valores son negativos — físicamente posible en elementos
#   fuera de la diagonal de Z0_barra cuando transformadores Dyn bloquean
#   la secuencia cero e invierten el sentido de la transferencia.
#   Usados en la fórmula exacta de tensión de fase:
#     V_a_mk = 1 - (2*Z1_mk + Z0_mk) / (2*Z1_kk + Z0_kk)
#   reemplazando la derivación analítica vía Ec. 3.40 (que usaba solo
#   secuencia positiva) y el parche Z1_eff_1f previo (que absorbía
#   el efecto de Z0_mk en un Z1_mk ficticio distinto al de la rama 3f).
# ===========================================================================

Z_MK_1 = {
    # ── Cardones 220 (bus 4) ─────────────────────────────────────────────
    (4,  1): 0.0580, (4,  2): 0.0646, (4,  3): 0.0710, (4,  4): 0.0774,
    (4,  5): 0.0437, (4,  6): 0.0311, (4,  7): 0.0235, (4,  8): 0.0070,
    (4,  9): 0.0022, (4, 10): 0.0012,
    # ── Alto Jahuel 500 (bus 15) ─────────────────────────────────────────
    (15,  9): 0.004752, (15, 10): 0.0022, (15, 11): 0.0047,
    (15, 12): 0.0113,  (15, 13): 0.0029, (15, 14): 0.0105,
    (15, 15): 0.0143,  (15, 16): 0.0098, (15, 17): 0.0057,
    (15, 18): 0.0033,  (15, 19): 0.0075, (15, 20): 0.0047,
    (15, 21): 0.00047, (15, 22): 0.0,    (15, 23): 0.0,
    # ── Valdivia 220 (bus 24) ────────────────────────────────────────────
    (24, 12): 0.0019, (24, 15): 0.0019, (24, 16): 0.0033,
    (24, 19): 0.0045, (24, 20): 0.0069, (24, 21): 0.0023,
    (24, 22): 0.0292, (24, 23): 0.0295, (24, 24): 0.0683, (24, 25): 0.0398,
}

Z_MK_0 = {
    # ── Cardones 220 (bus 4) ─────────────────────────────────────────────
    (4,  1): -0.0038, (4,  2): -0.0024, (4,  3):  0.0021, (4,  4):  0.0227,
    (4,  5):  0.0013, (4,  6): -0.0083, (4,  7): -0.0031, (4,  8): -0.0059,
    # ── Alto Jahuel 500 (bus 15) ─────────────────────────────────────────
    (15,  9): -0.003398, (15, 10): -0.000873, (15, 11):  0.0016,
    (15, 12):  0.0047,   (15, 13): -0.0011,   (15, 14):  0.0067,
    (15, 15):  0.0123,   (15, 16):  0.0014,   (15, 17):  0.00082,
    (15, 18): -0.0011,   (15, 19): -0.000074, (15, 20): -0.000092,
    (15, 21): -0.0008,   (15, 22): -0.00449,  (15, 23): -0.00451,
    # ── Valdivia 220 (bus 24) ────────────────────────────────────────────
    (24, 16): -0.0025, (24, 19): -0.0016, (24, 20): -0.00065,
    (24, 21): -0.0013, (24, 22):  0.0014, (24, 23):  0.0016,
    (24, 24):  0.0366, (24, 25):  0.0012,
}

# V_SIM_1F — tensiones de fase real simuladas (PowerFactory), derivadas
# analíticamente de Z_MK_1 y Z_MK_0 mediante la fórmula exacta:
#   V_a_mk = 1 - (2*Z1_mk + Z0_mk) / (2*Z1_kk + Z0_kk)
# Se mantiene como referencia de validación y para los pares sin Z0_mk.
# Solo se incluyen valores ≤ 0.90 p.u.
V_SIM_1F = {
    # ── Cardones 220 (bus 4) ─────────────────────────────────────────────
    (4,  1): 0.806, (4,  2): 0.723, (4,  3): 0.641, (4,  4): 0.000,
    (4,  5): 0.228, (4,  6): 0.750, (4,  7): 0.742,
    # ── Alto Jahuel 500 (bus 15) ─────────────────────────────────────────
    (15, 10): 0.888, (15, 11): 0.749, (15, 12): 0.456,
    (15, 14): 0.298, (15, 15): 0.000, (15, 16): 0.560,
    (15, 17): 0.815, (15, 19): 0.654, (15, 20): 0.711,
    # ── Valdivia 220 (bus 24) ────────────────────────────────────────────
    (24, 19): 0.821, (24, 20): 0.594, (24, 22): 0.538,
    (24, 23): 0.532, (24, 24): 0.000, (24, 25): 0.654,
}


# ===========================================================================
# Tensiones V_kj — tensión en barra k ante falla en barra j (extremos línea)
# Permite calcular Z_kj = (1-V_kj) × Z_jj  (simetría: Z_kj = Z_jk)
# ===========================================================================
V_KJ = {
    (1,  2): 0.0000, (2,  3): 0.0000, (3,  4): 0.0827, (4,  5): 0.0000,
    (5,  6): 0.5040, (6,  7): 0.2750, (7,  8): 0.5090, (8,  9): 0.1870,
    (9, 10): 0.0500, (10,11): 0.4290, (11,13): 0.4690, (13,14): 0.7160,
    (12,15): 0.2098, (12,16): 0.3010, (15,16): 0.2794, (16,19): 0.2520,
    (17,18): 0.5710, (20,21): 0.8280, (20,22): 0.8560, (20,23): 0.8520,
    (23,24): 0.5681, (24,25): 0.5697,
}


def get_zkj_simulacion(Z1kk_sim: list) -> dict:
    """
    Calcula Z_kj para cada par extremo de línea desde tensiones simuladas.
    Z_kj = (1 - V_kj) × Z_jj_sim  (Ec. 3.8 inversa, simétrica)
    Retorna dict {(k,j): Z_kj_pu} con claves en ambos sentidos.
    """
    zkj = {}
    for (k, j), vkj in V_KJ.items():
        z = (1.0 - vkj) * Z1kk_sim[j - 1]
        zkj[(k, j)] = z
        zkj[(j, k)] = z
    return zkj


def get_zkk_simulacion():
    """
    Z1_kk = Sbase / Scc3        (Ec. 3.33 inversa)
    Z0_kk = Sbase/Scc1 - 2·Z1  (Ec. 3.34 inversa)
    Datos exactos para las 25 barras — sin aproximaciones.
    """
    Z1kk = [SBASE / s for s in SCC3_MVA]
    Z0kk = [SBASE / SCC1_MVA[k] - 2*Z1kk[k] for k in range(NBUS)]
    return Z1kk, Z0kk


# ===========================================================================
# Cuadro 4.1: líneas de transmisión
# (fb, tb, X1[Ω/km], X0[Ω/km], L[km], V[kV])  — un circuito por fila
# ===========================================================================
_LINEAS = [
    (1,  2, 0.3998,1.4839,185.00,220),  #  0  Paposo – D. Almagro       c1
    (1,  2, 0.3998,1.4839,185.00,220),  #  1                             c2
    (2,  3, 0.3932,1.3088, 72.15,220),  #  2  D. Almagro – C. Pinto
    (3,  4, 0.3977,1.3227, 75.30,220),  #  3  Carrera Pinto – Cardones
    (4,  5, 0.4064,1.3227,132.70,220),  #  4  Cardones – Maitencillo     c1
    (4,  5, 0.4064,1.3227,133.30,220),  #  5                             c2
    (4,  5, 0.4064,1.3227,133.30,220),  #  6                             c3
    (5,  6, 0.3912,1.3114,112.60,220),  #  7  Maitencillo – P. Colorado  c1
    (5,  6, 0.3912,1.3114,112.60,220),  #  8                             c2
    (6,  7, 0.3912,1.3114, 84.00,220),  #  9  P. Colorado – Pan Azúcar   c1
    (6,  7, 0.3912,1.3114, 84.00,220),  # 10                             c2
    (7,  8, 0.3900,1.3114,228.00,220),  # 11  Pan Azúcar – Los Vilos     c1
    (7,  8, 0.3900,1.3114,228.00,220),  # 12                             c2
    (8,  9, 0.3938,1.3159, 97.10,220),  # 13  Los Vilos – Nogales        c1
    (8,  9, 0.3938,1.3159, 97.10,220),  # 14                             c2
    (9, 10, 0.3938,1.3159, 27.00,220),  # 15  Nogales – Quillota         c1
    (9, 10, 0.3938,1.3159, 27.00,220),  # 16                             c2
    (10,11, 0.2370,1.1185, 49.58,220),  # 17  Quillota – Polpaico 220    c1
    (10,11, 0.2370,1.1185, 49.58,220),  # 18                             c2
    (11,13, 0.4044,1.3903, 29.80,220),  # 19  Polpaico – Cerro Navia     c1
    (11,13, 0.4044,1.3903, 29.80,220),  # 20                             c2
    (13,14, 0.3496,1.3253, 39.20,220),  # 21  Cerro Navia – AJ 220       c1
    (13,14, 0.3496,1.3253, 39.20,220),  # 22                             c2
    (12,15, 0.2745,1.2314, 71.89,500),  # 23  Polpaico 500 – AJ 500
    (12,16, 0.2774,1.0653,309.59,500),  # 24  Polpaico 500 – Ancoa 500 (directo)
    (15,16, 0.3351,1.0404,240.49,500),  # 25  AJ 500 – Ancoa 500 (NUEVO)
    (16,19, 0.3339,1.0714,182.83,500),  # 26  Ancoa 500 – Charrúa 500    c1
    (16,19, 0.3310,1.0690,196.14,500),  # 27                             c2
    (17,18, 0.3840,1.2052, 65.00,220),  # 28  Ancoa 220 – Itahue         1 circuito
    (20,21, 0.3869,1.3435, 71.80,220),  # 29  Charrúa 220 – Concepción
    (20,22, 0.3955,1.3558,195.70,220),  # 30  Charrúa 220 – Temuco
    (20,23, 0.2900,1.2920,204.00,220),  # 31  Charrúa 220 – Cautín       c1
    (20,23, 0.2900,1.2920,204.00,220),  # 32                             c2
    (22,23, 0.3975,1.3711,  3.00,220),  # 33  Temuco – Cautín            c1
    (22,23, 0.3975,1.3711,  3.00,220),  # 34                             c2
    (23,24, 0.4058,1.3795,149.12,220),  # 35  Cautín – Valdivia          c1
    (23,24, 0.4058,1.3795,149.12,220),  # 36                             c2
    (24,25, 0.3978,1.3703,207.04,220),  # 37  Valdivia – Puerto Montt    c1
    (24,25, 0.3978,1.3703,207.04,220),  # 38                             c2
]

# Cuadro 4.2: transformadores (fb, tb, X1_pu, X0_pu)
_XFMR = [
    (11,12,0.0218,0.0148),(14,15,0.0216,0.0147),
    (16,17,0.0218,0.0148),(19,20,0.0202,0.0202),
]


def _xpu(x_ohm_km, L_km, V_kv):
    zb = ZBASE_220 if V_kv == 220 else ZBASE_500
    return x_ohm_km * L_km / zb


def _make_data():
    lpu = [(fb,tb,_xpu(x1,L,V),_xpu(x0,L,V)) for (fb,tb,x1,x0,L,V) in _LINEAS]
    xpu = list(_XFMR)

    # Árbol generador (id=1) — añade cada barra exactamente una vez
    # Índices referenciados a la nueva _LINEAS (39 entradas):
    #   0:1-2c1  2:2-3  3:3-4  4:4-5c1  7:5-6c1  9:6-7c1  11:7-8c1
    #  13:8-9c1  15:9-10c1  17:10-11c1  19:11-13c1  21:13-14c1
    #  xpu[0]:11→12  23:12→15  24:12→16  xpu[2]:16→17  28:17→18
    #  26:16→19c1  xpu[3]:19→20  29:20→21  30:20→22  31:20→23c1
    #  35:23→24c1  37:24→25c1
    tree = [
        lpu[0],  lpu[2],  lpu[3],  lpu[4],  lpu[7],  lpu[9],  lpu[11], lpu[13],
        lpu[15], lpu[17], lpu[19], lpu[21], xpu[0],  lpu[23], lpu[24],
        xpu[2],  lpu[28], lpu[26], xpu[3],  lpu[29], lpu[30], lpu[31],
        lpu[35], lpu[37],
    ]

    # Mallas (id=-1) — circuitos paralelos y enlace de cierre 15-16 (nuevo)
    #   1:1-2c2  5:4-5c2  6:4-5c3  8:5-6c2  10:6-7c2  12:7-8c2  14:8-9c2
    #  16:9-10c2  18:10-11c2  20:11-13c2  22:13-14c2  xpu[1]:14-15
    #  25:15-16(NUEVO)  27:16-19c2  32:20-23c2  33:22-23c1  34:22-23c2
    #  36:23-24c2  38:24-25c2
    mesh = [
        lpu[1],  lpu[5],  lpu[6],  lpu[8],  lpu[10], lpu[12], lpu[14],
        lpu[16], lpu[18], lpu[20], lpu[22], xpu[1],  lpu[25],
        lpu[27], lpu[32], lpu[33], lpu[34], lpu[36], lpu[38],
    ]
    src = [(0,1,0.001,0.001)]
    data1, data0 = [], []
    for (fb,tb,x1,x0) in src:
        data1.append([fb,tb,x1, 0]); data0.append([fb,tb,x0, 0])
    for (fb,tb,x1,x0) in tree:
        data1.append([fb,tb,x1, 1]); data0.append([fb,tb,x0, 1])
    for (fb,tb,x1,x0) in mesh:
        data1.append([fb,tb,x1,-1]); data0.append([fb,tb,x0,-1])
    return np.array(data1,dtype=float), np.array(data0,dtype=float)


DATA1, DATA0 = _make_data()


def get_corredores():
    from collections import defaultdict
    grupos = defaultdict(list)
    for (fb,tb,x1,x0,L,V) in _LINEAS:
        key = (min(fb,tb), max(fb,tb))
        grupos[key].append((_xpu(x1,L,V), _xpu(x0,L,V), L))
    corredores = []
    for (fb,tb), ents in sorted(grupos.items()):
        x1_tot = 1.0 / sum(1.0/e[0] for e in ents)   # paralelo — para Z_barra
        x0_tot = 1.0 / sum(1.0/e[1] for e in ents)
        x1_circ = ents[0][0]   # un circuito individual — para z_kj serie en Ec. 3.16
        x0_circ = ents[0][1]
        corredores.append({
            'from': fb, 'to': tb,
            'label': f'{NOMBRES[fb]} — {NOMBRES[tb]}',
            'x1_pu':      x1_tot,   # equivalente paralelo
            'x0_pu':      x0_tot,
            'x1_pu_circ': x1_circ,  # circuito individual (z_kj serie)
            'x0_pu_circ': x0_circ,
            'longitud_km': ents[0][2]
        })
    return corredores


if __name__ == '__main__':
    Z1kk, Z0kk = get_zkk_simulacion()
    print(f"{'Bus':>4}  {'Nombre':<24}  {'Scc3[MVA]':>10}  "
          f"{'Scc1[MVA]':>10}  {'Z1kk':>8}  {'Z0kk':>8}")
    for k in range(NBUS):
        print(f"{k+1:>4}  {NOMBRES[k+1]:<24}  {SCC3_MVA[k]:>10.2f}  "
              f"{SCC1_MVA[k]:>10.2f}  {Z1kk[k]:>8.5f}  {Z0kk[k]:>8.5f}")
    print(f"\nCorredores: {len(get_corredores())}")
