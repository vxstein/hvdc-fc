# hvdc-fc

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21709850.svg)](https://doi.org/10.5281/zenodo.21709850)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Estimación probabilística de fallas de conmutación en enlaces HVDC-LCC causadas por variaciones de tensión.

Implementación en Python 3 de la metodología desarrollada en la memoria de título del mismo nombre (Ingeniería Civil Eléctrica, Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile).

## Qué hace

Estima la tasa anual de fallas de conmutación (FC) en la barra inversora de un enlace HVDC con conmutación natural, acoplando dos componentes:

- **Criterio de susceptibilidad del inversor:** la probabilidad de FC en función de la tensión residual en la barra inversora, obtenida por simulación dinámica en SimPowerSystems.
- **Estimación probabilística de hundimientos de tensión:** la frecuencia anual con que las fallas de cortocircuito producen hundimientos de cada magnitud, calculada mediante el teorema de compensación y la matriz de impedancias de barra, tratando la posición de falla en línea como variable aleatoria uniforme.

Se aplica a un modelo simplificado de 25 barras del Sistema de Transmisión Nacional chileno, con tres barras hipotéticas de inversión: Cardones 220 kV, Alto Jahuel 500 kV y Valdivia 220 kV.

## Estructura

| Archivo | Contenido |
|---|---|
| `sic_datos.py` | Parámetros del sistema: líneas, transformadores, potencias de cortocircuito y datos de simulación |
| `zbarra.py` | Construcción de la matriz de impedancias de barra y su calibración |
| `cortocircuito.py` | Cálculo y verificación de potencias de cortocircuito |
| `tensiones_falla.py` | Tensiones de falla en barras y en puntos interiores de línea |
| `graficos.py` | Histogramas, curvas acumuladas y estimación de la tasa de FC |
| `main_sic.py` | Programa principal |
| `SIC_HVDC_v1_0.ipynb` | Notebook que reproduce el análisis completo |
| `resultados/salida_programa.txt` | Salida completa del programa |

## Uso

```bash
pip install numpy matplotlib
python main_sic.py
```

Alternativamente, abra `SIC_HVDC_v1_0.ipynb` y ejecute las celdas en orden. El notebook escribe los módulos en disco y luego los ejecuta, de modo que funciona sin instalación previa en Google Colab.

## Resultados

Tasas anuales estimadas de fallas de conmutación:

| Barra de inversión | FC/año |
|---|---|
| Cardones 220 kV | 4,40 |
| Alto Jahuel 500 kV | 5,88 |
| Valdivia 220 kV | 4,39 |

Entre el 90 % y el 93 % de la tasa estimada proviene de fallas monofásicas a tierra.

## Notas sobre esta versión

Corrige tres defectos de la versión anterior:

1. La matriz de impedancias de secuencia cero se construye desde `DATA0`. Anteriormente era una copia de la de secuencia positiva.
2. El cálculo monofásico sobre corredores de línea se restringe a tramos con impedancia de transferencia homopolar simulada en ambos extremos. Interpolar entre un extremo calibrado y otro algorítmico mezclaba escalas incompatibles y producía perfiles de tensión inadmisibles.
3. El truncamiento de tensión a [0, 1] emite aviso en vez de operar en silencio.

**Limitación conocida.** No se dispone de un valor calibrado de la impedancia de transferencia homopolar entre extremos de corredor, `Z0_kj`. El código adopta `Z0_kj = Z1_kj`, dimensionalmente incorrecto pero de escala compatible. La solución requiere una simulación de cortocircuito monofásico por corredor.

## Dependencias

`numpy` y `matplotlib`. No se emplean librerías especializadas de sistemas de potencia.

## Cita

VR (2026). *Estimación probabilística de fallas de conmutación en enlaces HVDC causadas por variaciones de tensión* [Memoria de título]. Universidad de Chile.
