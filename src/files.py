import csv
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def extraer_datos_sesion(ruta_archivo):
    tree = ET.parse(ruta_archivo)
    root = tree.getroot()
    static = root.find(".//static")
    
    p = {}

    # Función interna para buscar y extraer sin errores
    def buscar_texto(id_label):
        nodo = static.find(f"text[@IDlabel='{id_label}']")
        return nodo.get('data') if nodo is not None else "No encontrado"

    # Ahora extraes de forma segura
    p['Nombre'] = buscar_texto('First name') # 
    p['Apellido'] = buscar_texto('Last name') # 
    p['Fecha_Nacimiento'] = buscar_texto('Birthday') # 
    p['Sexo'] = buscar_texto('Sex') # 
    p['Fecha_Sesion'] = buscar_texto('Date session') # [cite: 2]

    # Para los valores numéricos con escala
    def buscar_numerico(tag, label, scale_default):
        nodo = static.find(f"{tag}[@label='{label}']")
        if nodo is not None:
            data = float(nodo.get('data'))
            scale = float(nodo.get('scaleFactor', scale_default))
            return data / scale
        return 0.0

    p['Masa_kg'] = buscar_numerico('mass', 'mTB', 1000) # [cite: 3]
    p['Altura_m'] = buscar_numerico('track1d', 'dTH', 10000) # [cite: 4]

    return p


def guardar_datos_matrices(matrices, nombres_matrices, archivo_csv):
    """
    Guarda los datos de 7 matrices (4, n, 3) en un archivo CSV.
    
    Parámetros:
    - matrices: Lista de 7 matrices numpy de forma (4, n, 3).
    - archivo_csv: Nombre del archivo CSV donde se guardarán los datos.
    """
    # Abrimos el archivo CSV en modo de escritura
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        header = []
        for nombre in nombres_matrices:
            header.extend([f'{nombre}_origen_x', f'{nombre}_origen_y', f'{nombre}_origen_z',
                           f'{nombre}_v1_x', f'{nombre}_v1_y', f'{nombre}_v1_z',
                           f'{nombre}_v2_x', f'{nombre}_v2_y', f'{nombre}_v2_z',
                           f'{nombre}_v3_x', f'{nombre}_v3_y', f'{nombre}_v3_z'])
        writer.writerow(header)
        
        # Asumimos que todas las matrices tienen el mismo número de frames
        n_frames = matrices[0].shape[1]  # El número de frames es el mismo en todas las matrices
        
        # Iteramos sobre cada frame
        for i in range(n_frames):
            # Lista para almacenar la fila de datos para el frame i
            fila = []
            
            # Iteramos sobre las 7 matrices para extraer los datos de cada una
            for m in matrices:
                # Extraemos el punto de origen (m[0, i, :])
                origen = m[0, i, :]
                # Extraemos los tres vectores (m[1:4, i, :])
                vector1 = m[1, i, :]
                vector2 = m[2, i, :]
                vector3 = m[3, i, :]
                
                # Añadimos los datos a la fila
                fila.extend(origen)       # Coordenadas del origen
                fila.extend(vector1)      # Primer vector
                fila.extend(vector2)      # Segundo vector
                fila.extend(vector3)      # Tercer vector
            
            # Escribimos la fila de datos en el archivo CSV
            writer.writerow(fila)

def save_euler_angles_to_csv(euler_angles, filename="euler_angles.csv", include_frame=True):
    """
    Guarda los ángulos de Euler en un CSV con:
    - Separador de columnas: ;
    - Separador decimal: ,

    :param euler_angles: Array de forma (n, 3) con los ángulos [theta_z, theta_y, theta_x] en grados.
    :param filename: Nombre del archivo CSV (por defecto: "euler_angles.csv").
    :param include_frame: Si True, añade una columna con el número de frame.
    """
    euler_angles = np.asarray(euler_angles)
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')  # Separador de columnas ;
        
        # Encabezado
        header = []
        if include_frame:
            header.append("frame")
        header.extend(["theta_z (deg)", "theta_y (deg)", "theta_x (deg)"])
        writer.writerow(header)
        
        # Datos (convertir puntos decimales a comas)
        for i, angles in enumerate(euler_angles):
            row = []
            if include_frame:
                row.append(i)
            # Formatear números: 1.23 → "1,23"
            row.extend([f"{angle:.6f}".replace('.', ',') for angle in angles])
            writer.writerow(row)
    
    print(f"Ángulos guardados en {filename} (separador: ;, decimal: ,)")

def save_marker_to_csv(marker_data, filename="marker.csv", include_frame=True):
    """
    Guarda los ángulos de Euler en un CSV con:
    - Separador de columnas: ;
    - Separador decimal: ,

    :param euler_angles: Array de forma (n, 3) con los ángulos [theta_z, theta_y, theta_x] en grados.
    :param filename: Nombre del archivo CSV (por defecto: "euler_angles.csv").
    :param include_frame: Si True, añade una columna con el número de frame.
    """
    marker_data = np.asarray(marker_data)
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')  # Separador de columnas ;
        
        # Encabezado
        header = []
        if include_frame:
            header.append("frame")
        header.extend(["x (m)", "y (m)", "z (m)"])
        writer.writerow(header)
        
        # Datos (convertir puntos decimales a comas)
        for i, markers in enumerate(marker_data):
            row = []
            if include_frame:
                row.append(i)
            # Formatear números: 1.23 → "1,23"
            row.extend([f"{mark:.6f}".replace('.', ',') for mark in markers ])
            writer.writerow(row)
    
    print(f"Marcador guardado en {filename} (separador: ;, decimal: ,)")   


def exportar_crudos_wide(data_sujeto_cond, archivo="ciclos_crudos.csv"):
    """
    Exporta ciclos crudos en formato wide.
    Parámetro esperado: data_sujeto_cond = data[sujeto][condicion]
    """
    registros = {}

    for lado, segmentos in data_sujeto_cond.items():
        for seg, planos in segmentos.items():
            for plano, contenido in planos.items():
                col_name = f"{lado}_{seg}_{plano}"
                ciclos = contenido["ciclos_crudos"]
                # Apilar todos los ciclos en una sola columna
                registros[col_name] = [valor for ciclo in ciclos for valor in ciclo]

    df = pd.DataFrame(registros)
    df.index.name = "index"
    df.to_csv(archivo)
    return df


def exportar_normalizados_wide(data_sujeto_cond, archivo="ciclos_normalizados.csv"):
    """
    Exporta ciclos normalizados en formato wide.
    Parámetro esperado: data_sujeto_cond = data[sujeto][condicion]
    """
    registros = {}

    for lado, segmentos in data_sujeto_cond.items():
        for seg, planos in segmentos.items():
            for plano, contenido in planos.items():
                col_name = f"{lado}_{seg}_{plano}"
                ciclos = contenido["ciclos_norm"]
                matriz = pd.DataFrame(ciclos).T  # (n_puntos+1, n_ciclos)
                registros[col_name] = matriz.mean(axis=1).values

    df = pd.DataFrame(registros)
    df.index.name = "index"
    df.to_csv(archivo)
    return df

