import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def distance_between_2_points(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p2) - np.array(p1))

def two_point_unit_vector_std(p1, p2):
    """Calcula el vector unitario desde el punto p1 al punto p2."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2 - p1
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Los puntos p1 y p2 son idénticos; el vector no puede ser definido.")
    return vector / norm

def two_point_unit_vector(p1, p2):
    """
    Calcula el vector unitario desde el punto p1 al punto p2.
    
    Parámetros:
    p1 (array-like): El primer punto, puede ser una lista, un array de coordenadas (3,) o (n, 3).
    p2 (array-like): El segundo punto, puede ser una lista, un array de coordenadas (3,) o (n, 3).

    Retorna:
    numpy.ndarray: El vector unitario desde p1 hasta p2. Si p1 y p2 tienen forma (n, 3), retorna (n, 3).

    Excepciones:
    ValueError: Si los puntos p1 y p2 no tienen la misma forma o si el vector no puede ser definido.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    

    if p1.shape != p2.shape:
        raise ValueError("Los puntos p1 y p2 deben tener la misma forma.")
    
    # Calcular el vector
    vector = p2 - p1
    
    # Calcular la norma
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    
    # Manejar el caso en que la norma es cero
    if np.any(norm == 0):
        raise ValueError("Al menos un par de puntos p1 y p2 son idénticos; el vector no puede ser definido.")
    
    # Dividir el vector por su norma para obtener el vector unitario
    return vector / norm

def point_uvector_distance(p1, uVector, dist):
    x = p1[0] + dist*uVector[0]
    y = p1[1] + dist*uVector[1]
    z = p1[2] + dist*uVector[2]
    return [x, y, z]

def punto_medio(p1, p2):
    """Calcula el punto medio entre dos puntos p1 y p2."""
    return (np.array(p1) + np.array(p2)) / 2.0

import numpy as np

def Point_Global_To_Local(pointG, rLocal, rGlobal):
    """
    Convierte un punto del sistema global al sistema local.
    
    Parámetros
    ----------
    pointG : ndarray de forma (3,)
        Coordenadas del punto en el sistema global.
    rLocal : ndarray de forma (4, 3)
        Sistema local: [origen, x_axis, y_axis, z_axis].
    rGlobal : ndarray de forma (4, 3)
        Sistema global: [origen, x_axis, y_axis, z_axis].

    Retorna
    -------
    pointL : ndarray de forma (3,)
        Coordenadas del punto en el sistema local.
    """

    # Extraer rotaciones (3x3) y orígenes
    R_local = rLocal[1:4, :].T     # cada columna es un eje del sistema local
    R_global = rGlobal[1:4, :].T   # cada columna es un eje del sistema global
    O_local = rLocal[0, :]
    O_global = rGlobal[0, :]

    # Transformación global -> local
    # Rotación relativa
    R_gl_to_loc = R_local.T @ R_global

    # Traslación relativa
    t_gl_to_loc = R_local.T @ (O_global - O_local)

    # Transformar el punto
    pointL = R_gl_to_loc @ pointG + t_gl_to_loc

    return pointL


def point_local_to_global(local_coords, local_frames, global_reference):
    """
    Transforma un punto de coordenadas locales a globales.

    :param local_coords: Coordenadas locales del punto de la forma (3,)
    :param local_frames: Matriz de transformación de la forma (4, n, 3) donde n es el número de frames.
                         La primera fila es el punto de origen, y las siguientes tres filas son los 
                         vectores unitarios que definen el sistema de coordenadas locales en cada frame.
    :param global_reference: Sistema de referencia global de la forma (4, 3).
                             La primera fila es el punto de origen, y las siguientes tres filas son 
                             los vectores unitarios que definen el sistema de coordenadas global.
    :return: Coordenadas globales del punto de la forma (n, 3)
    """
    if local_frames.shape == (4, 3):
        # Transformar de (4, 3) a (4, 1, 3)
        local_frames = np.expand_dims(local_frames, axis=1)
    
    n_frames = local_frames.shape[1]
    global_coords = np.zeros((n_frames, 3))

    for i in range(n_frames):
        local_origin = local_frames[0, i, :]
        local_rotation_matrix = local_frames[1:, i, :].T

        global_origin = global_reference[0, :]
        global_rotation_matrix = global_reference[1:, :].T

        # Primero, transforma las coordenadas locales al sistema global
        global_coords_local = np.dot(local_rotation_matrix, local_coords) + local_origin

        # Luego, ajusta las coordenadas transformadas al sistema de referencia global
        global_coords[i, :] = np.dot(global_rotation_matrix, global_coords_local - global_origin)

    return global_coords


def angle_between_2_unit_vectors(uVector1, uVector2):
    # Calcular la magnitud (norma) de los vectores
    norm_uVector1 = np.linalg.norm(uVector1, axis=1)
    norm_uVector2 = np.linalg.norm(uVector2, axis=1)
    
    # Calcular el producto punto entre los dos vectores
    dot_product = np.einsum('ij,ij->i', uVector1, uVector2)
    
    # Calcular el ángulo en radianes y luego convertirlo a grados
    cos_theta = dot_product / (norm_uVector1 * norm_uVector2)
    
    # Asegurarse de que el valor esté dentro del dominio [-1, 1] debido a posibles errores de precisión numérica
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calcular el ángulo en radianes y luego convertirlo a grados
    angles = np.degrees(np.arccos(cos_theta))
    
    return angles


def get_module(vector):
    return np.linalg.norm(vector)

def proyectar_en_plano(vec, vecPl1, vecPl2):
    n = np.cross(vecPl2, vecPl1)
    return np.cross(np.cross(n, vec), n)

def get_normal_vector(vec):
    if vec.ndim == 2:
        # Casos de múltiples pares de vectores (forma (n, 3))
        # Calcular la norma de cada vector
        vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
       
         # Verificar si hay vectores colineales (norma cero)
        if np.any(vec_norm == 0):
            raise ValueError("Al menos un par de vectores son colineales; el producto vectorial no puede ser definido.")
    
    elif vec.ndim == 1:
        # Caso de un solo par de vectores (forma (3,))
        # Calcular la norma del vector
        vec_norm = np.linalg.norm(vec)
        
        # Verificar si el vector es nulo (norma cero)
        if vec_norm == 0:
            raise ValueError("Los vectores son colineales; el producto vectorial no puede ser definido.")

    # Normalizar los vectores
    vec = vec / vec_norm
    return vec

def get_reference(mk1, mk2, mk3):
    """
    Calcula un sistema de coordenadas local basado en tres puntos de referencia mk1, mk2 y mk3.
    
    Parameters:
    mk1, mk2, mk3: arrays de coordenadas (x, y, z) de los puntos de referencia.
    
    Returns:
    array: una matriz (4x3) donde la primera fila es el origen, la segunda fila es el eje u,
           la tercera fila es el eje v, y la cuarta fila es el eje w.
    """
    # Cálculo del punto medio entre mk1 y mk2
    o_point = punto_medio(mk1, mk2)
    
    # Cálculo del vector unitario desde mk1 a mk2
    v = two_point_unit_vector(mk1, mk2)
    # Cálculo del vector unitario desde mk3 al punto medio
    u_temp = two_point_unit_vector(mk3, o_point)
    # Cálculo del vector u ortogonal a v y v_temp
    w = np.cross(u_temp, v)
    # Normalización de w
    w = get_normal_vector(w)
    # Cálculo del vector w ortogonal a u y v
    u = np.cross(v, w)
    # Normalización de u
    u = get_normal_vector(u)
       
    reference_matrix = np.array([o_point, u, v, w])
    
    return reference_matrix

def get_reference_form_vectors(mk1, mk2, mk3, mk4):
    """
    Calcula un sistema de coordenadas local basado en tres puntos de referencia mk1, mk2 (vector 1) y mk3, mk4 (vector 2).
    Vector 1 y 2 son los vectroes que generan el plan principal del sistema de referencia
    
    Parameters:
    mk1, mk2, mk3, mk4: arrays de coordenadas (x, y, z) de los puntos de referencia.
    
    Returns:
    array: una matriz (4x3) donde la primera fila es el origen, la segunda fila es el eje u,
           la tercera fila es el eje v, y la cuarta fila es el eje w.
    """
    # Cálculo del punto medio entre mk1 y mk2
    o_point = punto_medio(mk1, mk2)
    
    # Cálculo del vector unitario desde mk1 a mk2
    z = two_point_unit_vector(mk1, mk2)
    # Cálculo del vector unitario desde mk3 al punto medio
    y_temp = two_point_unit_vector(mk3, mk4)
    # Cálculo del vector u ortogonal a v y v_temp
    x = np.cross(y_temp, z)
    # Normalización de w
    x = get_normal_vector(x)
    # Cálculo del vector w ortogonal a u y v
    y = np.cross(z, x)
    # Normalización de u
    y = get_normal_vector(y)
       
    reference_matrix = np.array([o_point, x, y, z])
    
    return reference_matrix

"""
def get_reference_anatomical(mk1, mk2, mk3, leftside):
    
    Calcula un sistema de coordenadas local basado en tres puntos de referencia mk1, mk2 y mk3.
    en este caso mk1 y mk2 deben formar el eje principal y mk1 y mk3 el eje secundario . 

    Parameters:
    mk1, mk2, mk3: arrays de coordenadas (x, y, z) de los puntos de referencia.
    
    Returns:
    array: una matriz (4x3) donde la primera fila es el origen, la segunda fila es el eje u,
           la tercera fila es el eje v, y la cuarta fila es el eje w.
    
    # Cálculo del punto medio entre mk1 y mk2
    o_point = punto_medio(mk1, mk2)
    
    # Cálculo del vector unitario desde mk1 a mk2/ eje primario
    x = two_point_unit_vector(mk1, mk2)
    # Cálculo del vector unitario desde mk1 a mk3/ eje secundario
    z_temp = two_point_unit_vector(mk3, mk1)
    if leftside==1:
      z_temp = two_point_unit_vector(mk1, mk3)  
    # Cálculo del vector u ortogonal a v y v_temp
    y = np.cross(z_temp, x)
    # Normalización de w
    y = get_normal_vector(y)
    # Cálculo del vector w ortogonal a u y v
    z = np.cross(x, y)
    # Normalización de u
    z = get_normal_vector(z)
       
    reference_matrix = np.array([o_point, x, y, z])
    
    return reference_matrix

"""

def get_reference_anatomical(mk1, mk2, mk3, leftside):
    """
    Calcula un sistema de coordenadas local basado en tres puntos de referencia mk1, mk2 y mk3.
    En este caso mk1 y mk2 deben formar el eje principal y mk1 y mk3 el eje secundario. 

    Parameters:
    mk1, mk2, mk3: arrays de coordenadas (x, y, z) de los puntos de referencia.
    leftside: 1 si es el lado izquierdo, 0 si es el lado derecho.
    
    Returns:
    array: una matriz (4x3) donde la primera fila es el origen, la segunda fila es el eje u,
           la tercera fila es el eje v, y la cuarta fila es el eje w.
    """
    
    # Asegurar que los puntos son arrays 3D
    mk1 = np.asarray(mk1)
    mk2 = np.asarray(mk2)
    mk3 = np.asarray(mk3)

    # Verificar que los puntos son diferentes
    if np.all(mk1 == mk2) or np.all(mk1 == mk3):
        raise ValueError("Los puntos mk1, mk2 y mk3 no pueden ser iguales.")
    
    # Cálculo del punto medio entre mk1 y mk2
    o_point = punto_medio(mk1, mk2)
    
    # Cálculo del vector unitario desde mk1 a mk2 (eje primario)
    x = two_point_unit_vector(mk1, mk2)
    
    # Cálculo del vector unitario desde mk1 a mk3 (eje secundario)
    z_temp = two_point_unit_vector(mk3, mk1)
    
    if leftside == 1:
        z_temp = two_point_unit_vector(mk1, mk3)
    
    # Cálculo del vector y ortogonal a x y z_temp
    y = np.cross(z_temp, x)
    
    # Normalización de y
    y = get_normal_vector(y)
    
    # Cálculo del vector w ortogonal a x y
    z = np.cross(x, y)
    
    # Normalización de w
    z = get_normal_vector(z)
    
    # Matriz de referencia con los vectores de la base
    reference_matrix = np.array([o_point, x, y, z])

    
    return reference_matrix

    
def get_pointed_virtual_marker(p1, p2, lPointer):
    vPDIR = two_point_unit_vector(p2, p1)
    return point_uvector_distance(p1, vPDIR, lPointer)

def distance_between_2_points(p1, p2):
    return np.linalg.norm(p1-p2)

def point_from_2_points_and_scalar(p1, p2, scalar):
    vUnit = two_point_unit_vector(p1, p2)
    distance = distance_between_2_points(p1, p2)
    distance = scalar*distance
    return point_uvector_distance(p1, vUnit, distance)

def write_trc_header(n_markers, n_frames=1):
    f = open('Graphs/Standing.trc', 'w')
    title = 'PathFileType	4	(X/Y/Z)	2838~aa~Standing.trc\n'
    header = 'DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames\n'
    values = f'100.0	100.0	1	{n_markers}\tm	100.0	1	{n_frames}\n'
    marker_names = 'Frame#	Time	r_asis			l_asis			sacrum_s			r_hip			l_hip			r_bar_1			l_bar_1			r_knee_1			r_knee_local			l_knee_1			l_knee_local			r_ankle         l_ankle\n'
    positions = ''.join([f'X{n}\tY{n}\tZ{n}\t' for n in range(1, n_markers+1)])
    f.write(title + header + values + marker_names + '\t\t'+positions)
    return f

def write_trc_file(number_of_markers, markers):
    f = write_trc_header(number_of_markers)
    for i, val in enumerate(markers):
        values = ''.join([f'{number}\t' for number in val])
        f.write(f'{i}\t{i}\t{values}\n')

def rotate_90_deg_x(reference_frame):
    """
    Rota un sistema de referencia 90 grados sobre el eje X.
    
    :param reference_frame: Matriz de forma (4, n, 3) representando el sistema de referencia.
    :return: Matriz de forma (4, n, 3) con el sistema de referencia rotado.
    """
    # Matriz de rotación sobre el eje X
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Crear una copia del sistema de referencia para rotar
    rotated_frame = reference_frame.copy()

    # Aplicar la rotación a cada frame
    for i in range(reference_frame.shape[1]):
        rotated_frame[1:, i, :] = np.dot(rotation_matrix_x, reference_frame[1:, i, :])

    return rotated_frame

def rotate_90_deg_z(reference_frame):
    """
    Rota un sistema de referencia 90 grados sobre el eje z.
    
    :param reference_frame: Matriz de forma (4, n, 3) representando el sistema de referencia.
    :return: Matriz de forma (4, n, 3) con el sistema de referencia rotado.
    """
    # Matriz de rotación sobre el eje X
    rotation_matrix_x = np.array([
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Crear una copia del sistema de referencia para rotar
    rotated_frame = reference_frame.copy()

    # Aplicar la rotación a cada frame
    for i in range(reference_frame.shape[1]):
        rotated_frame[1:, i, :] = np.dot(rotation_matrix_x, reference_frame[1:, i, :])

    return rotated_frame    


def rotate_270_deg_z(reference_frame):
    """
    Rota un sistema de referencia 270 grados sobre el eje z (rotación en sentido horario).
    
    :param reference_frame: Matriz de forma (4, n, 3) representando el sistema de referencia.
    :return: Matriz de forma (4, n, 3) con el sistema de referencia rotado.
    """
    # Matriz de rotación de -90 grados (equivalente a 270 grados en sentido horario) sobre el eje Z
    rotation_matrix_z_270 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Crear una copia del sistema de referencia para rotar
    rotated_frame = reference_frame.copy()

    # Aplicar la rotación a cada frame
    for i in range(reference_frame.shape[1]):
        rotated_frame[1:, i, :] = np.dot(rotation_matrix_z_270, reference_frame[1:, i, :].T).T

    return rotated_frame

def rotate_270_deg_y(reference_frame):
    """
    Rota un sistema de referencia 270 grados sobre el eje Y (rotación en sentido antihorario).
    
    :param reference_frame: Matriz de forma (4, n, 3) representando el sistema de referencia.
    :return: Matriz de forma (4, n, 3) con el sistema de referencia rotado.
    """
    # Matriz de rotación de -90 grados sobre el eje Y (equivalente a 270 grados en sentido antihorario)
    rotation_matrix_y_270 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    # Crear una copia del sistema de referencia para rotar
    rotated_frame = reference_frame.copy()

    # Aplicar la rotación a cada frame
    for i in range(reference_frame.shape[1]):
        rotated_frame[1:, i, :] = np.dot(rotation_matrix_y_270, reference_frame[1:, i, :].T).T

    return rotated_frame

def change_sign_angles(L_lista, x_sign, y_sign, z_sign):
    """
    Cambia el signo de los ángulos theta_x y theta_y, dejando theta_z sin cambios.
    
    :param L_lista: Lista de arrays con los ángulos [theta_x, theta_y, theta_z] para cada frame.
    :return: Lista de arrays R_lista con los ángulos modificados.
    """
    # Inicializamos la lista donde guardaremos los nuevos ángulos
    R_lista = []
    
    # Iteramos sobre cada array en L_lista
    for angles in L_lista:
        # Cambiamos el signo de theta_x y theta_y
        theta_x, theta_y, theta_z = angles
        
        # Creamos el nuevo array con los signos cambiados para theta_x y theta_y
        new_angles = np.array([x_sign*theta_x, y_sign*theta_y, z_sign*theta_z])
        
        # Agregamos el nuevo array a la lista R_lista
        R_lista.append(new_angles)
    
    return R_lista


def rotate_by_270_degrees(point):
    # point[1] = [point[1][1], -point[1][0], point[1][2]]
    rotated = np.zeros((3,3))
    rotated[0] = -point[2]
    rotated[1] = point[1]
    rotated[2] = point[3]
    return rotated

def get_time_vector(frequency, signal):
    return [str(n/frequency) for n in range(signal.nFrames)]

def get_events(events):
    return { "RHS": np.round(events.events[0].values,2), "RTO": events.events[1].values, "LHS": np.round(events.events[2].values,2), "LTO": events.events[3].values}

def rotate_by_x(theta, point):
    rotation = [[ 1, 0           , 0           ],
                   [ 0, np.cos(np.radians(theta)),-np.sin(np.radians(theta))],
                   [ 0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]]
    return [np.array(i) for i in np.matmul(rotation, point).tolist()]

def rotate_by_y(theta, point):
    theta = math.radians(theta)
    rotation = [[ np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
                   [ 0           , 1, 0           ],
                   [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]]
    return [np.array(i) for i in np.matmul(rotation, point).tolist()]

def rotate_by_z(theta, point):
    theta = math.radians(theta)
    rotation = [[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return [np.array(i) for i in np.matmul(rotation, point).tolist()]

def get_rgait(r_heel, l_heel, vertical_lab_axis):
    x = two_point_unit_vector(r_heel, l_heel)
    y = np.cross(x, vertical_lab_axis)
    z = np.cross(y, vertical_lab_axis)
    return [x, y, z]

def to_degrees(angles):
    return [[math.degrees(angle[0]), math.degrees(angle[1]), math.degrees(angle[2])] for angle in angles]

def get_hip_rotation_center_harrington(leg_length, pelvis_depth, pelvis_width):

    x = -0.24*pelvis_depth - 9.9
    y = -(0.28*pelvis_depth + 0.16*pelvis_width + 7.9)
    z = -0.16 * pelvis_width - 0.04*leg_length - 7.1

    return [x, y, z]

def get_hip_rotation_center_hara(leg_length):
# DOI: 10.1038/srep37707
# Predicting the location of the hip joint centres, impact of age group and sex
# X: anterior-posterior Y: medio-lateral Z:Inferior - Superior

    x = (11 - 0.063 * leg_length)/1000
    y = ( 8 + 0.086 * leg_length)/1000
    z = (-9 - 0.078 * leg_length)/1000

    return [x, y, z] , [x,-y,z] 

def get_euler_angles(scl_pri, scl_sec):
    """
    Calcula los ángulos de Euler entre dos sistemas de referencia usando productos cruz y punto.
    
    :param scl_pri: Matriz de forma (4, n, 3) representando el sistema de referencia primario.
    :param scl_sec: Matriz de forma (4, n, 3) representando el sistema de referencia secundario.
    :return: Una lista de arrays con los ángulos [theta_x, theta_y, theta_z] para cada frame.
    """
    # Verificar si el sistema primario o secundario tiene un solo frame y expandirlo si es necesario
    if scl_pri.shape == (4, 3):
        n_frames = scl_sec.shape[1]
        scl_pri = np.tile(scl_pri[:, np.newaxis, :], (1, n_frames, 1))
    else:
        n_frames = scl_pri.shape[1]

    if scl_sec.shape == (4, 3):
        n_frames = scl_pri.shape[1]
        scl_sec = np.tile(scl_sec[:, np.newaxis, :], (1, n_frames, 1))

    angles = []

    for i in range(n_frames):
        # Excluir el origen (primera fila)
        scl_pri_frame = scl_pri[1:, i, :]
        scl_sec_frame = scl_sec[1:, i, :]

        # Línea de nodos vN
        vN = np.cross(scl_pri_frame[2], scl_sec_frame[1])

        # Cálculo de los ángulos de Euler
        theta_x = np.arcsin(np.dot(scl_pri_frame[2], scl_sec_frame[1]))
        theta_y = np.arcsin(np.dot(vN, scl_sec_frame[2]))
        theta_z = np.arcsin(np.dot(vN, scl_pri_frame[1]))

        angles.append([theta_x, theta_y, theta_z])

    return np.degrees(np.array(angles))
import numpy as np

def get_euler_angles_new(scl_pri, scl_sec):
    """
    Calcula los ángulos de Euler entre dos sistemas de referencia usando productos cruz y punto.

    :param scl_pri: Matriz de forma (4, n, 3) representando el sistema de referencia primario.
    :param scl_sec: Matriz de forma (4, n, 3) representando el sistema de referencia secundario.
    :return: Una lista de arrays con los ángulos [theta_x, theta_y, theta_z] para cada frame.
    """
    # Verificar y expandir si es necesario
    if scl_pri.shape == (4, 3):
        scl_pri = np.tile(scl_pri[:, np.newaxis, :], (1, scl_sec.shape[1], 1))
    if scl_sec.shape == (4, 3):
        scl_sec = np.tile(scl_sec[:, np.newaxis, :], (1, scl_pri.shape[1], 1))

    angles = []

    for i in range(scl_pri.shape[1]):
        # Excluir el origen (primera fila)
        scl_pri_frame = scl_pri[1:, i, :]
        scl_sec_frame = scl_sec[1:, i, :]

        # Línea de nodos vN
        vN = np.cross(scl_pri_frame[2], scl_sec_frame[1])

        # Cálculo de los ángulos de Euler
        theta_x = np.arctan2(np.dot(scl_sec_frame[1], scl_pri_frame[2]), np.dot(scl_sec_frame[0], scl_pri_frame[2]))
        theta_y = np.arctan2(np.dot(scl_pri_frame[0], vN), np.dot(scl_pri_frame[1], vN))
        theta_z = np.arctan2(np.dot(vN, scl_pri_frame[1]), np.dot(vN, scl_sec_frame[2]))

        angles.append([theta_x, theta_y, theta_z])

    return np.degrees(np.array(angles))

import numpy as np

def get_euler_angles_rota(scl_pri, scl_sec):
    """
    Calcula los ángulos de Euler entre dos sistemas de referencia usando matrices de rotación.

    :param scl_pri: Matriz de forma (4, n, 3) representando el sistema de referencia primario.
    :param scl_sec: Matriz de forma (4, n, 3) representando el sistema de referencia secundario.
    :return: Una lista de arrays con los ángulos [theta_x, theta_y, theta_z] para cada frame.
    """
    def rotation_matrix_to_euler_angles(R):
        """
        Convierte una matriz de rotación en ángulos de Euler.
        
        :param R: Matriz de rotación de forma (3, 3).
        :return: Ángulos de Euler [theta_x, theta_y, theta_z].
        """
        assert (R.shape == (3, 3))
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])
    
    def get_rotation_matrix_from_vectors(vec1, vec2):
        """
        Obtiene una matriz de rotación que alinea vec1 con vec2.
        
        :param vec1: Vector 1.
        :param vec2: Vector 2.
        :return: Matriz de rotación de forma (3, 3).
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        v = np.cross(vec1, vec2)
        c = np.dot(vec1, vec2)
        s = np.linalg.norm(v)
        I = np.eye(3)
        V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = I + V + V @ V * ((1 - c) / (s ** 2))
        return R

    # Verificar y expandir si es necesario
    if scl_pri.shape == (4, 3):
        scl_pri = np.tile(scl_pri[:, np.newaxis, :], (1, scl_sec.shape[1], 1))
    if scl_sec.shape == (4, 3):
        scl_sec = np.tile(scl_sec[:, np.newaxis, :], (1, scl_pri.shape[1], 1))

    angles = []

    for i in range(scl_pri.shape[1]):
        # Excluir el origen (primera fila)
        scl_pri_frame = scl_pri[1:, i, :]
        scl_sec_frame = scl_sec[1:, i, :]

        # Obtener las matrices de rotación entre los sistemas de referencia
        R = get_rotation_matrix_from_vectors(scl_pri_frame[:, 0], scl_sec_frame[:, 0])
        euler_angles = rotation_matrix_to_euler_angles(R)
        angles.append(euler_angles)

    return np.degrees(np.array(angles))


def expand_reference_matrix(reference_matrix, num_frames):
    """
    Expande una matriz de referencia de forma (4, 3) a (4, n, 3) para replicar el sistema de referencia en todos los frames.

    :param reference_matrix: Matriz de referencia de forma (4, 3).
    :param num_frames: Número de frames a replicar.
    :return: Matriz expandida de forma (4, n, 3).
    """
    if reference_matrix.shape != (4, 3):
        raise ValueError("La matriz de referencia debe tener la forma (4, 3).")
    
    return np.tile(reference_matrix[:, np.newaxis, :], (1, num_frames, 1))


def get_euler_angles_rot(scl_pri, scl_sec):
    """
    Calcula los ángulos de Euler entre dos sistemas de referencia usando matrices de rotación.

    :param scl_pri: Matriz de forma (4, n, 3) representando el sistema de referencia primario.
    :param scl_sec: Matriz de forma (4, n, 3) representando el sistema de referencia secundario.
    :return: Una lista de arrays con los ángulos [theta_x, theta_y, theta_z] para cada frame.
    """
    
    # Verificar dimensiones
    if scl_pri.shape != scl_sec.shape:
        raise ValueError("Las dimensiones de scl_pri y scl_sec deben coincidir.")
    
    if scl_pri.shape[0] != 4 or scl_sec.shape[0] != 4:
        raise ValueError("Las matrices deben tener la forma (4, n, 3).")
    
    # Expandir dimensiones si es necesario
    if scl_pri.shape[1] == 3:
        scl_pri = np.tile(scl_pri[:, np.newaxis, :], (1, scl_sec.shape[1], 1))
    if scl_sec.shape[1] == 3:
        scl_sec = np.tile(scl_sec[:, np.newaxis, :], (1, scl_pri.shape[1], 1))
    
    angles = []

    for i in range(scl_pri.shape[1]):
        # Excluir el origen (primera fila)
        scl_pri_frame = scl_pri[1:, i, :]
        scl_sec_frame = scl_sec[1:, i, :]

        # Asegurarse de que las matrices de rotación están bien definidas
        if scl_pri_frame.shape[0] != 3 or scl_sec_frame.shape[0] != 3:
            raise ValueError("Cada frame debe contener 3 vectores de 3 dimensiones.")

        # Calcular la matriz de rotación que alinea scl_pri con scl_sec
        # Aquí asumimos que los vectores están alineados en el orden [X, Y, Z]
        R_matrix = np.dot(np.linalg.inv(scl_pri_frame), scl_sec_frame)

        # Convertir la matriz de rotación en ángulos de Euler (XYZ)
        rotation = R.from_matrix(R_matrix)
        euler_angles = rotation.as_euler('ZYX', degrees=True)

        angles.append(euler_angles)

    return np.array(angles)

def get_euler_angles_rot2(scl_pri, scl_sec):
    """
    Calcula los ángulos de Euler entre dos sistemas de referencia.
    :param scl_pri: Matriz (4, n, 3) con [origen, X, Y, Z] del sistema primario.
    :param scl_sec: Matriz (4, n, 3) con [origen, X, Y, Z] del sistema secundario.
    :return: Ángulos de Euler (ZYX) en grados para cada frame.
    """
    if scl_pri.shape != scl_sec.shape:
        raise ValueError("Las dimensiones de scl_pri y scl_sec deben coincidir.")
    
    angles = []
    for i in range(scl_pri.shape[1]):
        # Extraer ejes X, Y, Z (ignorando el origen)
        pri_X, pri_Y, pri_Z = scl_pri[1:4, i, :]
        sec_X, sec_Y, sec_Z = scl_sec[1:4, i, :]
        
        # Normalizar vectores (opcional pero recomendado)
        pri_X = pri_X / np.linalg.norm(pri_X)
        pri_Y = pri_Y / np.linalg.norm(pri_Y)
        pri_Z = pri_Z / np.linalg.norm(pri_Z)
        sec_X = sec_X / np.linalg.norm(sec_X)
        sec_Y = sec_Y / np.linalg.norm(sec_Y)
        sec_Z = sec_Z / np.linalg.norm(sec_Z)
        
        # Construir matrices de rotación (asegurar ortogonalidad)
        R_pri = np.column_stack([pri_X, pri_Y, pri_Z])
        R_sec = np.column_stack([sec_X, sec_Y, sec_Z])
        
        # Matriz de rotación relativa: R_sec = R_pri @ R_rel
        R_rel = R_pri.T @ R_sec  # Usar transpuesta en lugar de inversa
        
        # Calcular ángulos de Euler (intrínsecos: 'xyz' o extrínsecos: 'ZYX')
        euler_angles = R.from_matrix(R_rel).as_euler('XYZ', degrees=True)
        angles.append(euler_angles)
    
    return np.array(angles)



def get_joint_angles_old(e_distal, e_proximal):
    value = (np.dot(e_distal[2], e_proximal[1]))
    clamped_value = np.clip(value, -1, 1)
    theta_x = np.arcsin(clamped_value)
    value = (np.dot(e_distal[2], e_proximal[0]))/np.cos(theta_x)
    clamped_value = np.clip(value, -1, 1)
    theta_y = np.arcsin(clamped_value)
    value = (np.dot(e_distal[0], e_proximal[1]))/np.cos(theta_x)
    clamped_value = np.clip(value, -1, 1)
    theta_z =np.arcsin(clamped_value)

    return [theta_x, theta_y, theta_z]

def angulo_con_plano_sagital(vectores_3D, isLeft):
    # Definir el vector normal al plano sagital (0, 0, 1)
    normal_plano_sagital = np.array([0, 0, 1])
    
    # Inicializar una lista para los ángulos
    angulos = []
    
    # Iterar sobre los vectores 3D (en cada frame)
    for v in vectores_3D:
        # Calcular el producto escalar entre el vector 3D y el normal al plano
        producto_escalar = np.dot(v, normal_plano_sagital)
        
        # Calcular la magnitud del vector 3D
        magnitud_v = np.linalg.norm(v)
        
        # Calcular el coseno del ángulo entre el vector 3D y el normal
        cos_theta = producto_escalar / magnitud_v
        
        # Asegurarse de que el valor de cos_theta esté en el rango [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calcular el ángulo en radianes y convertirlo a grados
        angulo_rad = np.arccos(cos_theta)
        angulo_grados = np.degrees(angulo_rad)
        
        # El ángulo entre el vector y el plano es el complemento del ángulo con el eje z
        if isLeft ==1: 
            angulo_con_plano = angulo_grados - 90
        else: 
            angulo_con_plano = 90 - angulo_grados    
        
        # Agregar el ángulo al resultado
        angulos.append(angulo_con_plano)
    
    return angulos

def angulo_con_plano_frontal(vectores_3D, isLeft):
    # Definir el vector normal al plano frontal (0, 1, 0)
    normal_plano_frontal = np.array([0, 1, 0])
    
    # Inicializar una lista para los ángulos
    angulos = []
    
    # Iterar sobre los vectores 3D (en cada frame)
    for v in vectores_3D:
        # Calcular el producto escalar entre el vector 3D y el normal al plano
        producto_escalar = np.dot(v, normal_plano_frontal)
        
        # Calcular la magnitud del vector 3D
        magnitud_v = np.linalg.norm(v)
        
        # Calcular el coseno del ángulo entre el vector 3D y el normal
        cos_theta = producto_escalar / magnitud_v
        
        # Asegurarse de que el valor de cos_theta esté en el rango [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calcular el ángulo en radianes y convertirlo a grados
        angulo_rad = np.arccos(cos_theta)
        angulo_grados = np.degrees(angulo_rad)
        
        # El ángulo entre el vector y el plano es el complemento del ángulo con el eje x
        if isLeft ==1: 
            angulo_con_plano = angulo_grados - 90
        else: 
            angulo_con_plano = 90 - angulo_grados    
        
        # Agregar el ángulo al resultado
        angulos.append(angulo_con_plano)
    
    return angulos

def calcular_angulo_proyectado(v_3d, v_horiz, v_vert):
    # Normalizar ejes del plano (opcional pero recomendable)
    v_horiz = v_horiz / np.linalg.norm(v_horiz, axis=1, keepdims=True)
    v_vert = v_vert / np.linalg.norm(v_vert, axis=1, keepdims=True)

    # Vector normal al plano frontal
    v_normal = np.cross(v_horiz, v_vert)
    v_normal /= np.linalg.norm(v_normal, axis=1, keepdims=True)

    # Proyectar v_3d sobre el plano
    v_proj = v_3d - np.sum(v_3d * v_normal, axis=1, keepdims=True) * v_normal

    v_rearhor = np.cross(v_proj, v_normal)

    # Calcular el ángulo con respecto al eje vertical usando el producto escalar
    dot_product = np.einsum('ij,ij->i', v_rearhor, v_vert)
    norm_proj = np.linalg.norm(v_rearhor, axis=1)
    norm_vert = np.linalg.norm(v_vert, axis=1)

    cos_angle = dot_product / (norm_proj * norm_vert)
    angulo_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angulo_deg = np.degrees(angulo_rad)

    return angulo_deg

# Función para calcular ángulo proyectado con signo
def calcular_angulo_proyectado2(v_3d, v_horiz, v_vert):
    v_horiz_norm = v_horiz 
    v_vert_norm = v_vert 
    v_normal = np.cross(v_horiz_norm, v_vert_norm)
    v_normal /= np.linalg.norm(v_normal, axis=1)[:, np.newaxis]
    v_proj = v_3d - np.sum(v_3d * v_normal, axis=1)[:, np.newaxis] * v_normal
    dot_vert = np.einsum('ij,ij->i', v_proj, v_vert_norm)
    dot_horiz = np.einsum('ij,ij->i', v_proj, v_horiz_norm)
    angulo_rad = np.arctan2(dot_horiz, dot_vert)
    return np.degrees(angulo_rad)
