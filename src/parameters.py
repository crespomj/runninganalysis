import numpy as np

def calculate_running_metrics(pos_foot_L, pos_foot_R, pos_sacrum, 
                              ic_L, ic_R, to_L, to_R, 
                              v_cinta, fs):
    """
    Calcula métricas avanzadas de carrera en cinta.
    Orientación: Corredor mirando hacia -X.
    Ejes: X=Longitudinal, Z=Vertical.
    """
    
    # --- 1. TIEMPOS DE APOYO Y VUELO (Bilateral) ---
    def get_temporal_phases(ics, tos):
        apoyos = []
        vuelos = []
        for i in range(len(ics) - 1):
            # Stance (Apoyo): Desde contacto inicial hasta despegue
            # Buscamos el TO que ocurre inmediatamente después del IC
            current_to = tos[tos > ics[i]]
            if len(current_to) > 0 and current_to[0] < ics[i+1]:
                t_apoyo = (current_to[0] - ics[i]) / fs
                apoyos.append(t_apoyo)
                
                # Flight (Vuelo): Desde ese despegue hasta el siguiente contacto
                t_vuelo = (ics[i+1] - current_to[0]) / fs
                vuelos.append(t_vuelo)
        return apoyos, vuelos

    apoyos_L, vuelos_L = get_temporal_phases(ic_L, to_L)
    apoyos_R, vuelos_R = get_temporal_phases(ic_R, to_R)
    
    t_apoyo_avg = np.mean(apoyos_L + apoyos_R)
    t_vuelo_avg = np.mean(vuelos_L + vuelos_R)

    # --- 2. CADENCIA (Pasos por minuto) ---
    # Tiempo entre contactos sucesivos de ambos pies
    # Una forma robusta: 120 / promedio de duración de ciclos de un solo pie
    duracion_ciclos = np.diff(ic_L) / fs
    cadencia = 120 / np.mean(duracion_ciclos)

    # --- 3. LARGO DE CICLO Y PASO (Orientación -X) ---
    # Formula: (X_inicial - X_final) + (V_cinta * delta_t)
    strides = []
    for i in range(len(ic_L) - 1):
        dist_marcador = pos_foot_L[ic_L[i], 0] - pos_foot_L[ic_L[i+1], 0]
        dist_cinta = v_cinta * ((ic_L[i+1] - ic_L[i]) / fs)/3.6
        strides.append(dist_marcador + dist_cinta)
    
    largo_ciclo = np.mean(strides)
    largo_paso = largo_ciclo / 2 # Estimación asumiendo simetría

    # --- 4. OSCILACIÓN VERTICAL (Basada en Sacro) ---
    # Se mide el desplazamiento vertical del sacro dentro de cada ciclo de carrera
    oscilaciones = []
    for i in range(len(ic_L) - 1):
        segmento_y = pos_sacrum[ic_L[i] : ic_L[i+1], 1]
        oscilaciones.append(np.max(segmento_y) - np.min(segmento_y))
    
    oscilacion_vertical = np.mean(oscilaciones)

    # --- 5. VELOCIDAD ESTIMADA ---
    # Verificación: v = largo_ciclo * frecuencia_ciclo
    velocidad_est = largo_ciclo * (1 / np.mean(duracion_ciclos))

    return {
        "cadencia_ppm": round(cadencia, 2),
        "largo_ciclo_m": round(largo_ciclo, 3),
        "largo_paso_m": round(largo_paso, 3),
        "tiempo_apoyo_s": round(t_apoyo_avg, 3),
        "tiempo_vuelo_s": round(t_vuelo_avg, 3),
        "oscilacion_vertical_m": round(oscilacion_vertical, 3),
        "duty_factor_percent": round((t_apoyo_avg / (t_apoyo_avg + t_vuelo_avg)) * 100, 2),
        "velocidad_estimada_ms": round(velocidad_est, 2)
    }

def calculate_step_lengths_bilateral(pos_L, pos_R, ic_L, ic_R, v_cinta, fs):
    """
    Calcula el largo de paso individual para cada lado.
    Orientación: Corredor mirando hacia -X.
    """
    steps_L = [] # Distancia del paso que termina con el pie Izquierdo
    steps_R = [] # Distancia del paso que termina con el pie Derecho

    print(pos_L[0:20,0])

    # 1. Paso Izquierdo: Distancia desde IC Derecho hasta IC Izquierdo
    for t_R in ic_R:
        siguiente_ic_L = ic_L[ic_L > t_R]
        if len(siguiente_ic_L) > 0:
            t_L = siguiente_ic_L[0]
            dt = (t_L - t_R) / fs
            # Al mirar hacia -X, la distancia es (X_atrás - X_adelante)
            # En t_L, el pie L está adelante (menor X) y el R atrás (mayor X)
            dist_marcador = pos_R[t_R, 0] - pos_L[t_L, 0]
            steps_L.append(dist_marcador + (v_cinta * dt/3.6))

    # 2. Paso Derecho: Distancia desde IC Izquierdo hasta IC Derecho
    for t_L in ic_L:
        siguiente_ic_R = ic_R[ic_R > t_L]
        if len(siguiente_ic_R) > 0:
            t_R = siguiente_ic_R[0]
            dt = (t_R - t_L) / fs
            # En t_R, el pie R está adelante (menor X) y el L atrás (mayor X)
            dist_marcador = pos_L[t_L, 0] - pos_R[t_R, 0]
            steps_R.append(dist_marcador + (v_cinta * dt/3.6))

    return np.mean(steps_L), np.mean(steps_R)

def calculate_symmetry_index(val_L, val_R):
    """
    Calcula el Índice de Simetría (SI). 
    0% indica simetría perfecta.
    """
    if val_L == 0 or val_R == 0: return 0
    
    # Fórmula estándar de Robinson (1987)
    si = (abs(val_L - val_R) / (0.5 * (val_L + val_R))) * 100
    return si

# --- Función Integrada Final ---

def get_complete_report(pos_L, pos_R, pos_sacrum, ic_L, ic_R, to_L, to_R, v_cinta, fs):
    """Genera el reporte final con simetrías."""
    
    # Largos de paso reales
    lp_izq, lp_der = calculate_step_lengths_bilateral(pos_L, pos_R, ic_L, ic_R, v_cinta, fs)
    
    # Simetría de paso
    si_paso = calculate_symmetry_index(lp_izq, lp_der)
    
    # Tiempos de apoyo para simetría temporal
    t_apoyo_L = np.mean(np.diff(np.sort(np.concatenate((ic_L, to_L))))) # Simplificado para el ejemplo
    # (En tu código usarías la lógica de la respuesta anterior para mayor precisión)
    
    osc_L, osc_R = calculate_vertical_oscillation_bilateral(pos_sacrum, ic_L, ic_R)
    si_oscilacion = calculate_symmetry_index(osc_L, osc_R)

    # Agregar al reporte:
    # 

    return {
        "largo_paso_izq_m": round(lp_izq, 3),
        "largo_paso_der_m": round(lp_der, 3),
        "simetria_paso_porcentaje": round(si_paso, 2),
        "comentario": "Simétrico" if si_paso < 5 else "Asimetría detectada",
        "oscilacion_V_izq_cm": round(osc_L * 100, 2),
        "oscilacion_V_der_cm": round(osc_R * 100, 2),
        "simetria_oscilacion_V": round(si_oscilacion, 2)  
        
    }

def calculate_vertical_oscillation_bilateral(pos_sacrum, ic_L, ic_R):
    """
    Calcula la oscilación vertical del sacro desglosada por paso.
    Se define la oscilación del paso como el rango (max-min) del eje Z 
    durante el tiempo que dura dicho paso.
    """
    osc_paso_L = [] # Oscilación durante el paso izquierdo
    osc_paso_R = [] # Oscilación durante el paso derecho

    # 1. Oscilación en el paso izquierdo (Desde IC Derecho hasta IC Izquierdo)
    for t_R in ic_R:
        sig_ic_L = ic_L[ic_L > t_R]
        if len(sig_ic_L) > 0:
            # Extraemos el segmento vertical (eje Z) del sacro
            segmento_y = pos_sacrum[t_R : sig_ic_L[0], 1]
            if len(segmento_y) > 0:
                osc_paso_L.append(np.max(segmento_y) - np.min(segmento_y))

    # 2. Oscilación en el paso derecho (Desde IC Izquierdo hasta IC Derecho)
    for t_L in ic_L:
        sig_ic_R = ic_R[ic_R > t_L]
        if len(sig_ic_R) > 0:
            segmento_y = pos_sacrum[t_L : sig_ic_R[0], 1]
            if len(segmento_y) > 0:
                osc_paso_R.append(np.max(segmento_y) - np.min(segmento_y))

    return np.mean(osc_paso_L), np.mean(osc_paso_R)

