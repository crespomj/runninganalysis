import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detectar_eventos(kinematic_data, distancia_minima_frames=40, eje_vertical=1):
    """
    Detecta eventos de contacto inicial (heel strike) y despegue (toe off) para ambos pies
    utilizando los mínimos locales en las componentes verticales de los marcadores del talón y la punta del pie.

    Parámetros:
    - kinematic_data: dict con arrays {'talon_izq', 'talon_der', 'punta_izq', 'punta_der'} de forma (n_frames, 3)
    - distancia_minima_frames: separación mínima entre eventos detectados
    - eje_vertical: índice del eje vertical (1 = Y, 2 = Z según sistema de coordenadas)

    Retorna:
    - eventos: dict con claves 'izq' y 'der', cada una con listas de índices para 'contacto_inicial' y 'despegue'
    """
    eventos = {}

    for lado in ['izq', 'der']:
        talon_y = kinematic_data[f'talon_{lado}'][:, eje_vertical]
        punta_y = kinematic_data[f'punta_{lado}'][:, eje_vertical]

        # Mínimos locales en señal invertida → posiciones donde el marcador está más bajo
        contactos_iniciales, _ = find_peaks(-talon_y, distance=distancia_minima_frames)
        despegues, _ = find_peaks(-punta_y, distance=distancia_minima_frames)

        eventos[lado] = {
            'contactos_iniciales': contactos_iniciales,
            'despegues': despegues
        }

    return eventos

def valores_en_eventos(angulo, eventos, lado='der', tipo_evento='contactos_iniciales'):
    """
    Extrae los valores del ángulo en los frames correspondientes a un tipo de evento y lado específico.

    Parámetros:
    - angulo: array de forma (n,), con los valores del ángulo a lo largo del tiempo.
    - eventos: diccionario con eventos por lado, como el que retorna detectar_eventos().
    - lado: 'izq' o 'der' para seleccionar el pie izquierdo o derecho.
    - tipo_evento: 'contacto_iniciales' o 'despegues'.

    Retorna:
    - Lista con los valores del ángulo en los frames de los eventos seleccionados.
    """
    indices_eventos = eventos.get(lado, {}).get(tipo_evento, [])
    return [angulo[i] for i in indices_eventos if 0 <= i < len(angulo)]


def detectar_eventos_10_ciclos_centrales(
    kinematic_data, 
    distancia_minima_frames=40, 
    eje_vertical=1,
    offset_frames_izq=0,
    offset_frames_der=0
):
    from scipy.signal import find_peaks
    import numpy as np

    eventos_completos = {}
    eventos_filtrados = {}

    # Paso 1: Detectar todos los eventos para ambos pies
    for lado, offset in zip(['izq', 'der'], [offset_frames_izq, offset_frames_der]):
        talon_y = kinematic_data[f'talon_{lado}'][:, eje_vertical]
        punta_y = kinematic_data[f'punta_{lado}'][:, eje_vertical]

        contactos_iniciales, _ = find_peaks(-talon_y, distance=distancia_minima_frames)
        despegues, _ = find_peaks(-punta_y, distance=distancia_minima_frames)

        # Offset específico por lado
        contactos_iniciales = contactos_iniciales + offset
        despegues = despegues + offset

        eventos_completos[lado] = {
            'contactos_iniciales': contactos_iniciales,
            'despegues': despegues
        }

    # (resto de la función igual que antes...)
    contactos_izq = eventos_completos['izq']['contactos_iniciales']
    contactos_der = eventos_completos['der']['contactos_iniciales']

    n_ciclos_izq = len(contactos_izq) - 1
    n_ciclos_der = len(contactos_der) - 1

    if n_ciclos_izq < 10 or n_ciclos_der < 10:
        raise ValueError("No hay suficientes ciclos completos en ambos pies para extraer 10 ciclos centrales.")

    inicio_ciclo = (n_ciclos_izq - 10) // 2
    ciclos_centrales_izq = [(contactos_izq[i], contactos_izq[i+1]) for i in range(inicio_ciclo, inicio_ciclo + 10)]
    inicio_frame = ciclos_centrales_izq[0][0]
    fin_frame = ciclos_centrales_izq[-1][1]

    ciclos_centrales_der = []
    for i in range(len(contactos_der) - 1):
        if contactos_der[i] >= inicio_frame and contactos_der[i+1] <= fin_frame:
            ciclos_centrales_der.append((contactos_der[i], contactos_der[i+1]))

    if len(ciclos_centrales_der) < 10:
        for i in range(len(contactos_der) - 1):
            if contactos_der[i] >= fin_frame:
                ciclos_centrales_der.append((contactos_der[i], contactos_der[i+1]))

    if len(ciclos_centrales_der) < 10:
        raise ValueError("No hay suficientes ciclos completos del pie derecho dentro del rango de los 10 ciclos del pie izquierdo.")

    for lado in ['izq', 'der']:
        ci = eventos_completos[lado]['contactos_iniciales']
        dp = eventos_completos[lado]['despegues']

        ci_filtrados = ci[(ci >= inicio_frame) & (ci < fin_frame)]
        dp_filtrados = dp[(dp >= inicio_frame) & (dp < fin_frame)]

        eventos_filtrados[lado] = {
            'contactos_iniciales': ci_filtrados.tolist(),
            'despegues': dp_filtrados.tolist()
        }

    return eventos_filtrados, (inicio_frame, fin_frame)


def detectar_contactos_despegues(
    flexion_rodilla,
    thr_flexion_alta=80,
    thr_flexion_baja=20,
    distancia_minima_frames=20,
    plot=False
):
    """
    Detecta eventos de contacto inicial y despegue a partir de la curva de flexión de rodilla.

    Reglas:
    - Contacto: mínimo de extensión entre un máximo alto (>thr_flexion_alta) y el siguiente máximo bajo (<thr_flexion_baja).
    - Despegue: mínimo de extensión entre un máximo bajo (<thr_flexion_baja) y el siguiente máximo alto (>thr_flexion_alta).
    """

    flex = np.asarray(flexion_rodilla).ravel()

    # Detectar máximos y mínimos
    picos_flex, _ = find_peaks(flex, distance=distancia_minima_frames)
    picos_ext, _ = find_peaks(-flex, distance=distancia_minima_frames)

    contactos, despegues = [], []

    # Clasificar máximos
    max_altos = [p for p in picos_flex if flex[p] >= thr_flexion_alta]
    max_bajos = [p for p in picos_flex if flex[p] <= thr_flexion_baja]

    # Recorremos secuencias de picos de flexión
    for i in range(len(picos_flex) - 1):
        m1, m2 = picos_flex[i], picos_flex[i + 1]

        if flex[m1] >= thr_flexion_alta and flex[m2] <= thr_flexion_baja:
            # Contacto: buscar primer mínimo entre ellos
            candidatos = [p for p in picos_ext if m1 < p < m2]
            if candidatos:
                contactos.append(candidatos[0])

        elif flex[m1] <= thr_flexion_baja and flex[m2] >= thr_flexion_alta:
            # Despegue: buscar primer mínimo entre ellos
            candidatos = [p for p in picos_ext if m1 < p < m2]
            if candidatos:
                despegues.append(candidatos[0])

    # ---------- Gráfico opcional ----------
    if plot:
        plt.figure(figsize=(14, 5))
        plt.plot(flex, "k-", label="Flexión rodilla")

        plt.plot(picos_flex, flex[picos_flex], "ro", label="Picos flexión")
        plt.plot(picos_ext, flex[picos_ext], "bo", alpha=0.5, label="Mínimos extensión")

        plt.plot(contactos, flex[contactos], "go", markersize=10, label="Contactos")
        plt.plot(despegues, flex[despegues], "mo", markersize=10, label="Despegues")

        plt.legend()
        plt.title("Detección de Contactos y Despegues")
        plt.xlabel("Frame")
        plt.ylabel("Ángulo (°)")
        plt.show()

    return np.array(contactos, dtype=int), np.array(despegues, dtype=int)


def detectar_eventos_rodilla(flexion_rodilla, distancia_frames=20):
    flex = np.asarray(flexion_rodilla).ravel()

    # 1. Detectamos todos los picos de flexión sin filtrar por altura todavía
    picos_flex, propiedades = find_peaks(flex, distance=distancia_frames, prominence=5)
    alturas = flex[picos_flex]
    
    if len(picos_flex) < 2:
        return np.array([]), np.array([])

    # 2. Clasificamos picos en "Altos" (Balanceo) y "Bajos" (Apoyo)
    # Usamos el punto medio entre el máximo y el mínimo de los picos detectados
    umbral_decision = (np.max(alturas) + np.min(alturas)) / 2
    
    es_balanceo = alturas > umbral_decision
    es_apoyo = ~es_balanceo

    # 3. Detectamos mínimos (extensión)
    picos_ext, _ = find_peaks(-flex, distance=distancia_frames, prominence=2)
    
    contactos, despegues = [], []

    # 4. Buscamos eventos ciclo a ciclo
    for i in range(len(picos_flex) - 1):
        idx1, idx2 = picos_flex[i], picos_flex[i+1]
        
        # Caso A: De Balanceo a Apoyo -> Aquí está el CONTACTO INICIAL
        if es_balanceo[i] and es_apoyo[i+1]:
            candidatos = [p for p in picos_ext if idx1 < p < idx2]
            if candidatos:
                # El contacto es el pico de extensión justo antes de la flexión de apoyo
                contactos.append(candidatos[-1])
        
        # Caso B: De Apoyo a Balanceo -> Aquí está el DESPEGUE (TOE-OFF)
        elif es_apoyo[i] and es_balanceo[i+1]:
            candidatos = [p for p in picos_ext if idx1 < p < idx2]
            if candidatos:
                # El despegue es el pico de extensión justo después del apoyo
                despegues.append(candidatos[0])

    return np.array(contactos), np.array(despegues)

def recortar_ciclos_centrales_normalizados(angle, contactos, n_ciclos=10, n_puntos=100):
    """
    Recorta la señal de flexión de rodilla en los n ciclos centrales
    definidos entre eventos de contacto inicial, y normaliza cada ciclo
    a un número fijo de puntos (default=100).

    Parámetros:
    - flexion_rodilla: array de ángulos (flexión de rodilla).
    - contactos: array de índices (frames) de contactos iniciales.
    - n_ciclos: cantidad de ciclos a conservar (por defecto 10).
    - n_puntos: número de puntos para normalizar cada ciclo (default 100).

    Devuelve:
    - ciclos: lista de arrays con los ciclos crudos (longitud variable).
    - matriz_norm: matriz (n_ciclos × n_puntos) con los ciclos normalizados.
    - (inicio_frame, fin_frame): rango de frames que cubre los ciclos seleccionados.
    """

    contactos = np.asarray(contactos, dtype=int)

    if len(contactos) < n_ciclos + 1:
        raise ValueError(f"No hay suficientes contactos ({len(contactos)}) para formar {n_ciclos} ciclos.")

    # Construir todos los ciclos crudos
    ciclos = []
    for i in range(len(contactos) - 1):
        inicio, fin = contactos[i], contactos[i+1]
        ciclos.append(angle[inicio:fin])

    # Seleccionar ciclos centrales
    inicio_idx = (len(ciclos) - n_ciclos) // 2
    ciclos_centrales = ciclos[inicio_idx : inicio_idx + n_ciclos]

    inicio_frame = contactos[inicio_idx]
    fin_frame = contactos[inicio_idx + n_ciclos]

    # Normalizar cada ciclo a n_puntos
    matriz_norm = []
    for ciclo in ciclos_centrales:
        x_old = np.linspace(0, 1, len(ciclo))
        x_new = np.linspace(0, 1, n_puntos)
        ciclo_interp = np.interp(x_new, x_old, ciclo)
        matriz_norm.append(ciclo_interp)

    matriz_norm = np.vstack(matriz_norm)

    return ciclos_centrales, matriz_norm, (inicio_frame, fin_frame)


def armar_estructura_ciclos(
    angulos_por_segmento, 
    contactos_l, 
    contactos_r, 
    n_ciclos=10, 
    n_puntos=101, # Usar 101 permite tener del 0% al 100% exactos
    sujeto='sujeto1', 
    condicion='cond1'
):
    lados = angulos_por_segmento.keys()           
    planos = ['x','y','z']

    data = {sujeto: {condicion: {}}}

    for lado in lados:
        data[sujeto][condicion][lado] = {}
        contactos = contactos_l if lado == 'l' else contactos_r
        
        # VALIDACIÓN 1: ¿Hay suficientes contactos para armar al menos un ciclo?
        if len(contactos) < 2:
            print(f"⚠️ Aviso: No hay suficientes ciclos para el lado {lado}")
            continue

        # Ajustamos n_ciclos si hay menos de los pedidos
        ciclos_disponibles = len(contactos) - 1
        n_a_procesar = min(n_ciclos, ciclos_disponibles)

        # Solo iteramos sobre los segmentos que REALMENTE existen en los datos
        for seg in angulos_por_segmento[lado].keys():
            data[sujeto][condicion][lado][seg] = {}
            angulo_array = angulos_por_segmento[lado][seg]

            # VALIDACIÓN 2: Asegurar que el array no sea None o vacío
            if angulo_array is None: continue

            if angulo_array.ndim == 1:
                ciclos_crudos, ciclos_norm, (ini, fin) = recortar_ciclos_centrales_normalizados(
                    angulo_array, contactos, n_ciclos=n_a_procesar, n_puntos=n_puntos
                )
                data[sujeto][condicion][lado][seg]['x'] = {
                    'ciclos_crudos': ciclos_crudos, 'ciclos_norm': ciclos_norm,
                    'inicio_frame': ini, 'fin_frame': fin
                }
            else:
                for i, plano in enumerate(planos):
                    if i < angulo_array.shape[1]: # Evita error si el array tiene menos de 3 columnas
                        vector_plano = angulo_array[:, i]
                        ciclos_crudos, ciclos_norm, (ini, fin) = recortar_ciclos_centrales_normalizados(
                            vector_plano, contactos, n_ciclos=n_a_procesar, n_puntos=n_puntos
                        )
                        data[sujeto][condicion][lado][seg][plano] = {
                            'ciclos_crudos': ciclos_crudos, 'ciclos_norm': ciclos_norm,
                            'inicio_frame': ini, 'fin_frame': fin
                        }
    return data