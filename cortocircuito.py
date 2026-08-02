"""
cortocircuito.py — Potencias de Cortocircuito Trifásica y Monofásica
=====================================================================
Ecuaciones de la memoria (VR, 2026):

  Scc3φ = 1 / Z1_kk              (Ec. 3.33)  [p.u.] → × Sbase [MVA]
  Scc1φ = 1 / (2·Z1_kk + Z0_kk) (Ec. 3.34)  [p.u.] → × Sbase [MVA]

donde Z1_kk y Z0_kk son los elementos diagonales de las matrices de
impedancias de secuencia positiva y cero, respectivamente (= impedancias
equivalentes de Thévenin en la barra k).
"""

import numpy as np


def calcular_potencias_cc(
    Z1: np.ndarray,
    Z0: np.ndarray,
    Sbase: float = 100.0,
    nombres: dict = None,
) -> list[dict]:
    """
    Calcula Scc3φ y Scc1φ para cada barra.

    Parámetros
    ----------
    Z1, Z0 : ndarray (nbus, nbus), complex
        Matrices de impedancias de barra de secuencia positiva y cero.
    Sbase : float
        Potencia base en MVA (default 100 MVA).
    nombres : dict
        Mapa {numero_barra: nombre_barra}.

    Retorna
    -------
    Lista de dicts con claves:
      barra, nombre, Z1kk_pu, Z0kk_pu, X1kk_pu, X0kk_pu,
      Scc3_MVA, Scc1_MVA, Icc3_pu, Icc1_pu
    """
    nbus = Z1.shape[0]
    resultados = []

    for k in range(nbus):
        Z1kk = Z1[k, k]
        Z0kk = Z0[k, k]

        # Potencias de cortocircuito (Ecs. 3.33 y 3.34)
        Scc3 = Sbase / abs(Z1kk)          # MVA
        Scc1 = Sbase / abs(2*Z1kk + Z0kk) # MVA

        # Corrientes de cortocircuito en p.u. (con V_pref = 1 p.u.)
        Icc3 = 1.0 / abs(Z1kk)
        Icc1 = 3.0 / abs(2*Z1kk + Z0kk)  # corriente de secuencia cero × 3

        resultados.append({
            'barra'   : k + 1,
            'nombre'  : nombres.get(k + 1, f'Bus {k+1}') if nombres else f'Bus {k+1}',
            'Z1kk_pu' : Z1kk,
            'Z0kk_pu' : Z0kk,
            'X1kk_pu' : Z1kk.imag,
            'X0kk_pu' : Z0kk.imag,
            'Scc3_MVA': Scc3,
            'Scc1_MVA': Scc1,
            'Icc3_pu' : Icc3,
            'Icc1_pu' : Icc1,
        })

    return resultados


def imprimir_tabla_cc(resultados: list[dict], Sbase: float = 100.0) -> None:
    """Imprime tabla de resultados de cortocircuito."""
    print(f"\n{'='*80}")
    print("  POTENCIAS DE CORTOCIRCUITO — STN 25 barras")
    print(f"  Base: Sbase = {Sbase:.0f} MVA")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'Barra':<24}  {'X1kk':>8}  {'X0kk':>8}  "
          f"{'Scc3 [MVA]':>11}  {'Scc1 [MVA]':>11}")
    print(f"  {'-'*3}  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*11}")

    for r in resultados:
        print(f"  {r['barra']:>3}  {r['nombre']:<24}  "
              f"{r['X1kk_pu']:>8.5f}  {r['X0kk_pu']:>8.5f}  "
              f"{r['Scc3_MVA']:>11.1f}  {r['Scc1_MVA']:>11.1f}")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sic_datos import DATA1, DATA0, NBUS, NOMBRES, SBASE
    from zbarra   import construir_zbarra

    Z1 = construir_zbarra(DATA1, NBUS)
    Z0 = construir_zbarra(DATA0, NBUS)
    res = calcular_potencias_cc(Z1, Z0, Sbase=SBASE, nombres=NOMBRES)
    imprimir_tabla_cc(res, Sbase=SBASE)
