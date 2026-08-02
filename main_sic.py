"""
main_sic.py — Análisis Completo STN 25 Barras (v1.0)
======================================================
Metodología: VR (2026)

Correcciones incorporadas respecto de las iteraciones previas de desarrollo:
  - Z_barra de secuencia cero construida desde DATA0. Anteriormente la
    matriz homopolar era una copia de la de secuencia positiva: DATA0 se
    importaba pero nunca se usaba en el flujo principal.
  - Filtro de cobertura en el cálculo monofásico sobre líneas: solo se
    evalúan corredores con Z0_mk simulado en ambos extremos. Sin este
    filtro se interpolaba entre un extremo calibrado y otro algorítmico,
    con escalas separadas por dos órdenes de magnitud, lo que producía
    mínimos interiores de ~0 p.u. en corredores cuyos dos extremos
    superaban 0,94 p.u.
  - El clamp de tensión a [0,1] emite aviso en vez de operar en silencio.
  - Corregido el nombre de la barra 6: Punta Colorada.

  NOTA. No se modifica la asignación Z0_kj = Z1_kj del término cruzado de
  Ec. 3.16. Es dimensionalmente incorrecta, pero no existe dato calibrado
  de Z0_kj y sustituirla por el valor algorítmico introduce un error mayor
  (las tasas caían de 4,40/5,88/4,39 a 3,08/1,10/1,09 FC/año). La solución
  definitiva es calibrarla con una simulación monofásica por corredor:
      Z0_kj = (1 - V_a,kj_sim)·(2·Z1_jj + Z0_jj) - 2·Z1_kj
  análoga a la Ec. 3.41 ya empleada en secuencia positiva.

Otras decisiones metodológicas adoptadas durante el desarrollo:
  - Elimina el parche Z1_eff_1f (calibrar_zmk_1f_simulado), que era
    metodológicamente inconsistente: asignaba dos valores distintos de
    Z1_mk según el tipo de falla, sin justificación física.
  - Incorpora Z_MK_0 (sic_datos.py): impedancias de transferencia de
    secuencia cero simuladas directamente en PowerFactory.
  - tension_falla_barra_1f y tension_falla_linea_1f usan la fórmula
    exacta de tres secuencias:
        V_a_mk = 1 - (2·Z1_mk + Z0_mk) / (2·Z1_kk + Z0_kk)
    con Z1_mk idéntico al de la rama 3φ (consistencia física) y Z0_mk
    de simulación directa para los pares disponibles.

Flujo:
  1. Z_barra base desde algoritmo (Apéndice B), secuencias positiva y cero
  2. Calibración con datos de simulación DIgSILENT:
       - Diagonal Z_kk: exacta desde Scc3 y Scc1 (todas las barras)
       - Off-diagonal Z1_mk: exacta desde V_sim_3f (Ec. 3.8 inversa)
       - Off-diagonal Z0_mk: exacta desde Z_MK_0
  3. Tensiones de falla en barras:  fórmula 3-secuencias con Z_MK_0
  4. Tensiones de falla en líneas:  ídem con interpolación de Z0_mp
  5. Áreas de vulnerabilidad
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from sic_datos import (
    DATA1, DATA0, NBUS, NOMBRES, BARRAS_OBS, SBASE, V_UMBRAL, V_NO_CRITICO,
    get_corredores, get_zkk_simulacion, get_zkj_simulacion,
    SCC3_MVA, SCC1_MVA, V_SIM_3F, V_SIM_1F, V_KJ,
    Z_MK_1, Z_MK_0,
)
from zbarra import construir_zbarra, imprimir_zbarra, calibrar_zbarra_sim
from cortocircuito import calcular_potencias_cc, imprimir_tabla_cc
from tensiones_falla import (
    tension_falla_barra_3f, tension_falla_barra_1f,
    tension_falla_linea_3f, tension_falla_linea_1f,
    areas_vulnerabilidad_barras, areas_vulnerabilidad_lineas,
    curvas_area_vulnerabilidad,
    imprimir_tensiones_barras, imprimir_tensiones_lineas,
    imprimir_areas_vulnerabilidad,
)
from graficos import generar_graficos, imprimir_tasas_fc


def run_analisis():
    """
    Ejecuta el análisis completo y retorna todos los resultados.
    Llamar desde notebook para luego generar gráficos inline.
    """
    print("\n" + "="*65)
    print("  ANALISIS DE FALLAS — STN SIMPLIFICADO 25 BARRAS  (v1.0)")
    print("  Metodologia: VR (2026) + Z_MK_0 simulado")
    print("="*65)
    print(f"  Sbase={SBASE:.0f} MVA | Barras:{NBUS} | Umbral V={V_UMBRAL} p.u.")

    print("\n[1/5] Construyendo Z_barra base (algoritmo Apendice B) ...")
    Z_algo   = construir_zbarra(DATA1, NBUS)   # secuencia positiva
    Z_algo_0 = construir_zbarra(DATA0, NBUS)   # secuencia cero

    print("[2/5] Calibrando Z_barra con datos de simulacion ...")
    Z1kk_sim, Z0kk_sim = get_zkk_simulacion()
    zkj_sim = get_zkj_simulacion(Z1kk_sim)
    Z1, Z0  = calibrar_zbarra_sim(Z_algo, Z1kk_sim, Z0kk_sim, V_SIM_3F,
                                  obs_buses=list(BARRAS_OBS.keys()),
                                  v_default=V_NO_CRITICO,
                                  Z_algo_0=Z_algo_0)
    print(f"  Diagonal Z_kk:        exacta desde Scc3/Scc1 (25 barras)")
    print(f"  Off-diagonal Z1_mk:   exacta desde V_sim_3f ({len(V_SIM_3F)} pares)")
    print(f"  Off-diagonal Z0_mk:   exacta desde Z_MK_0 ({len(Z_MK_0)} pares)")
    print(f"  Z_kj extremos linea:  exacta desde V_kj ({len(V_KJ)} lineas)")
    print(f"  Z_barra sec. cero:    algoritmo sobre DATA0")
    print(f"  Filtro cobertura 1f:  solo corredores con Z0_mk simulado en "
          f"ambos extremos")

    print("\n[3/5] Verificando potencias de cortocircuito ...")
    res_cc = calcular_potencias_cc(Z1, Z0, Sbase=SBASE, nombres=NOMBRES)
    imprimir_tabla_cc(res_cc, Sbase=SBASE)
    print(f"\n  Verificacion barras de observacion:")
    for barra, nombre in BARRAS_OBS.items():
        r = res_cc[barra - 1]
        print(f"    {nombre}: Scc3={r['Scc3_MVA']:.1f} MVA "
              f"(sim={SCC3_MVA[barra-1]:.1f})  |  "
              f"Scc1={r['Scc1_MVA']:.1f} MVA "
              f"(sim={SCC1_MVA[barra-1]:.1f})")

    print("\n[4/5] Calculando tensiones de falla ...")
    corredores = get_corredores()
    v3f  = tension_falla_barra_3f(Z1, BARRAS_OBS, V_SIM_3F)
    v1f  = tension_falla_barra_1f(Z1, Z0, BARRAS_OBS,
                                   z_mk_1=Z_MK_1, z_mk_0=Z_MK_0,
                                   v_sim_1f=V_SIM_1F, v_sim_3f=V_SIM_3F)
    vl3f = tension_falla_linea_3f(Z1, corredores, BARRAS_OBS,
                                   v_sim=V_SIM_3F, zkj_sim=zkj_sim,
                                   v_umbral=V_UMBRAL)
    vl1f = tension_falla_linea_1f(Z1, Z0, corredores, BARRAS_OBS,
                                   v_sim_3f=V_SIM_3F, v_sim_1f=V_SIM_1F,
                                   zkj_sim=zkj_sim, v_umbral=V_UMBRAL,
                                   z_mk_1=Z_MK_1, z_mk_0=Z_MK_0)
    imprimir_tensiones_barras(v3f, v1f, BARRAS_OBS, NOMBRES, V_UMBRAL)
    imprimir_tensiones_lineas(vl3f, vl1f, BARRAS_OBS, V_UMBRAL)

    print("\n[5/5] Identificando areas de vulnerabilidad ...")
    areas_b = areas_vulnerabilidad_barras(v3f, v1f, BARRAS_OBS, NOMBRES, V_UMBRAL)
    areas_l = areas_vulnerabilidad_lineas(vl3f, vl1f, BARRAS_OBS, V_UMBRAL)
    imprimir_areas_vulnerabilidad(areas_b, areas_l, BARRAS_OBS, NOMBRES)

    print(f"\n{'='*65}")
    print("  RESUMEN EJECUTIVO")
    print(f"{'='*65}")
    for m, nombre in BARRAS_OBS.items():
        b3 = areas_b[m]['3f'];  b1 = areas_b[m]['1f']
        l3 = areas_l[m]['3f'];  l1 = areas_l[m]['1f']
        print(f"\n  {nombre}:")
        print(f"    Barras criticas -> 3f: {len(b3)}  |  1f: {len(b1)}")
        print(f"    Lineas criticas -> 3f: {len(l3)}  |  1f: {len(l1)}")
    print(f"{'='*65}\n")

    curvas_obs = curvas_area_vulnerabilidad(
        vl3f, vl1f, areas_l, corredores, BARRAS_OBS
    )

    imprimir_tasas_fc(curvas_obs, areas_b, v3f, v1f, BARRAS_OBS, V_UMBRAL)

    return dict(
        Z1=Z1, Z0=Z0,
        v3f=v3f, v1f=v1f,
        vl3f=vl3f, vl1f=vl1f,
        areas_b=areas_b, areas_l=areas_l,
        curvas_obs=curvas_obs,
        corredores=corredores,
    )


def main():
    """Punto de entrada CLI — analisis + graficos."""
    res = run_analisis()
    print("\n[6/6] Generando graficos ...")
    generar_graficos(
        res['curvas_obs'], res['areas_b'], res['areas_l'],
        res['v3f'], res['v1f'],
        BARRAS_OBS, NOMBRES, V_UMBRAL
    )


if __name__ == '__main__':
    main()
