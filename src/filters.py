from scipy.signal import savgol_filter
import numpy as np

def suavizar_savgol(data, window_length=9, polyorder=2):
    """
    Suaviza un array de datos 3D usando el filtro de Savitzky-Golay.

    Parámetros:
    - data: array (n, 3)
    - window_length: debe ser impar, cantidad de frames por ventana
    - polyorder: orden del polinomio

    Retorna:
    - data_suavizada: array suavizado
    """
    if window_length % 2 == 0:
        raise ValueError("window_length debe ser impar.")

    data_suavizada = np.zeros_like(data)
    for i in range(3):
        data_suavizada[:, i] = savgol_filter(data[:, i], window_length, polyorder)

    return data_suavizada

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy import interpolate

def interpolar_datos(data, metodo='cubic'):
    """
    Interpola datos con valores faltantes (NaN) de forma eficiente y vectorizada.

    Parámetros
    ----------
    data : ndarray (n, 3) o (n, m)
        Datos con posibles valores NaN.
    metodo : str, opcional
        Método de interpolación: 'linear', 'quadratic', 'cubic', etc.

    Retorna
    -------
    data_interp : ndarray del mismo tamaño que data
        Datos con valores NaN reemplazados por la interpolación.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("El array de entrada debe ser bidimensional (n, m).")

    n, m = data.shape
    x = np.arange(n)
    data_interp = np.copy(data)

    for i in range(m):
        y = data[:, i]
        mask = np.isfinite(y)

        if np.count_nonzero(mask) < 2:
            # Si hay menos de dos valores válidos, no se puede interpolar
            continue

        f = interpolate.interp1d(x[mask], y[mask], kind=metodo, fill_value="extrapolate")
        data_interp[:, i] = f(x)

    return data_interp


def suavizar_butterworth(data, cutoff=10, fs=100, order=2):
    """
    Suaviza un array de datos 3D usando un filtro Butterworth pasabajo de fase cero.
    
    Parámetros:
    ----------
    data : array-like (n, 3)
        Señal a suavizar. Cada columna es una componente (x, y, z).
    cutoff : float, opcional
        Frecuencia de corte (Hz).
    fs : float, opcional
        Frecuencia de muestreo (Hz).
    order : int, opcional
        Orden del filtro.

    Retorna:
    -------
    data_filtrada : ndarray (n, 3)
        Señal suavizada.
    """

    data = np.asarray(data, dtype=np.float64)

    # Acepta (3, n) o (n, 3)
    if data.ndim != 2:
        raise ValueError("El array de entrada debe ser bidimensional.")
    if data.shape[0] == 3 and data.shape[1] != 3:
        data = data.T
    if data.shape[1] != 3:
        raise ValueError("El array de entrada debe tener forma (n, 3).")

    # Calcular coeficientes del filtro (en formato SOS)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')

    # Aplicar filtro Butterworth (vectorizado)
    data_filtrada = sosfiltfilt(sos, data, axis=0)

    return data_filtrada

def suavizar_butterworth_old(data, cutoff=10, fs=100, order=2):
    """
    Suaviza un array de datos 3D usando un filtro Butterworth pasabajo de fase cero.

    Parámetros:
    - data: array (n, 3)
    - cutoff: frecuencia de corte (Hz)
    - fs: frecuencia de muestreo (Hz)
    - order: orden del filtro

    Retorna:
    - data_filtrada: array suavizado
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("El array de entrada debe tener forma (n, 3).")

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    data_filtrada = np.zeros_like(data)
    for i in range(3):
        data_filtrada[:, i] = filtfilt(b, a, data[:, i])

    return data_filtrada
