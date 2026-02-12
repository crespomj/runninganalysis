import numpy as np
import pandas as pd
from basictdf import Tdf

# Importamos tus módulos de biomecánica
from src.functions import *
import src.files as fl
import src.events as ev
import src.filters as ft
import src.parameters as pm
import src.graphic as gr

class RunningProcessor:
    def __init__(self, rutas, callback_progreso, velocidad, longitud):
        self.rutas = rutas
        self.progreso = callback_progreso  # Esta función actualiza la barra en main.py
        self.velocidad_cinta = velocidad
        self.largo_pierna = longitud
        # Definimos el sistema de referencia global una sola vez
        self.sr_global = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.angulos={}

    def ejecutar(self):
        """Método principal que orquestra el análisis"""
        try:
            # 1. METADATA
            self.progreso(0.1, "Extrayendo metadata de la sesión...")
            metadata = fl.extraer_datos_sesion(self.rutas['mdx'])

            # 2. PROCESAR STANDING (Calibración)
            self.progreso(0.2, "Calculando centros de rotación (Standing)...")
            referencias_calib = self._analizar_standing()

            # 3. CARGAR Y FILTRAR RUNNING
            self.progreso(0.4, "Filtrando señales de carrera...")
            markers_rn = self._cargar_y_filtrar_running()
            print(markers_rn)

            # 4. CÁLCULOS BIOMECÁNICOS (Cinemática)
            self.progreso(0.6, "Calculando ángulos de Euler...")
            angulos = self._calcular_cinematica(markers_rn, referencias_calib)

            self.progreso(0.8, "Detectando eventos de marcha...")
            contactos_r, despegues_r = ev.detectar_eventos_rodilla(self.angulos['r']['knee'][:, 2],
            distancia_frames=20)

            contactos_l, despegues_l = ev.detectar_eventos_rodilla(self.angulos['l']['knee'][:, 2],
            distancia_frames=20)
            print("termina eventos")
        
            #CALCULO DE PARAMETROS
            parametros_dt = pm.calculate_running_metrics(markers_rn['l heel mk'], 
                                         markers_rn['r heel mk'], 
                                         markers_rn['sacrum'], 
                                         contactos_l, 
                                         contactos_r, 
                                         despegues_l, 
                                         despegues_r,
                                         self.velocidad_cinta,
                                         100 
                                    )
            print(parametros_dt)

            param_report= pm.get_complete_report(markers_rn['l heel mk'],
                                                 markers_rn['r heel mk'],
                                                 markers_rn['sacrum'],
                                                 contactos_l,
                                                 contactos_r,
                                                 despegues_l, 
                                                 despegues_r,
                                                 self.velocidad_cinta,
                                                 100)
            
            print(param_report)
            print("ok parametros")

                      
            data = ev.armar_estructura_ciclos(  angulos_por_segmento=self.angulos,
                                                contactos_l=contactos_l,
                                                contactos_r=contactos_r,
                                                n_ciclos=10,
                                                n_puntos=101
                                            )
            
            data_sujeto_cond = data["sujeto1"]["cond1"]

            # Exportar crudos (apilados)
            #df_crudos = fl.exportar_crudos_wide(data_sujeto_cond, archivo="ciclos_crudos.csv")

            # Exportar normalizados (promedio, 101 filas)
            #df_norm = fl.exportar_normalizados_wide(data_sujeto_cond, archivo="ciclos_normalizados.csv")

            gr.graficar_curvas(data_sujeto_cond, modo="ambos")


            self.progreso(1.0, "Análisis finalizado.")
            return True, "Proceso exitoso"

        except Exception as e:
            # Captura cualquier error y lo envía de vuelta a la interfaz
            return False, f"Error en el procesamiento: {str(e)}"

    def _analizar_standing(self):
        """Carga el TDF y calcula centros y sistemas de referencia usando un diccionario seguro."""
        with Tdf(self.rutas['standing']) as f:
            standing = f.data3D

        # 1. Definimos los marcadores que vamos a buscar
        marcadores_interes = [
            'r tp', 'l tp', 'sacrum', 'r asis', 'l asis', 
            'r knee', 'l knee', 'r knee m', 'l knee m',
            'r thigh', 'l thigh', 'r mall', 'l mall',
            'r mall m', 'l mall m', 'r shank', 'l shank',
            'r foot', 'l foot', 'r heel', 'l heel',
            'r met', 'l met', 'r met 1st', 'r met 5th',
            'l met 1st', 'l met 5th'
        ]

        # 2. Cargamos el diccionario 'medias' de forma segura
        medias = {}
        for name in marcadores_interes:
            try:
                # Intentamos calcular la media (X, Y, Z) como hacías en tu función m_mkr
                # Usamos np.nanmean para ignorar posibles cortes en la señal
                x = np.nanmean(standing[name].X)
                y = np.nanmean(standing[name].Y)
                z = np.nanmean(standing[name].Z)
                medias[name] = np.array([x, y, z])
            except (KeyError, AttributeError):
                # Si el marcador no existe en este TDF, le asignamos NaNs
                medias[name] = np.array([np.nan, np.nan, np.nan])
                print(f"⚠️ Aviso: Marcador '{name}' no encontrado en el archivo.")
        
        #centros de rotación de cadera en coordenadas locales de la pelvis
        l_hip_center, r_hip_center = get_hip_rotation_center_hara(self.largo_pierna)
        # Centro de referencia de la pelvis
        srt_pelvis_st = get_reference(np.array(medias['r asis']), np.array(medias['l asis']), 
                                      punto_medio(medias['r tp'], medias['l tp']))
        # Centro de rotaición de cadera en coordenadas globales
        r_hip_st = point_local_to_global(r_hip_center, srt_pelvis_st, self.sr_global)
        l_hip_st = point_local_to_global(l_hip_center, srt_pelvis_st, self.sr_global)
   
        sist_ref_tri = get_reference(np.array(medias['r tp']), np.array(medias['l tp']), np.array(medias['sacrum']))
        r_asis_local = Point_Global_To_Local(np.array(medias['r asis']), sist_ref_tri, self.sr_global)
        l_asis_local = Point_Global_To_Local(np.array(medias['l asis']), sist_ref_tri, self.sr_global)

        r_knee_center = punto_medio(medias['r knee'], medias['r knee m'])
        l_knee_center = punto_medio(medias['l knee'], medias['l knee m'])
        srt_r_thigh = get_reference(medias['r knee'], r_hip_st.reshape(-1), medias['r thigh'])
        srt_l_thigh = get_reference(medias['l knee'], l_hip_st.reshape(-1), medias['l thigh'])
        r_knee_rel = Point_Global_To_Local(np.array(r_knee_center), srt_r_thigh, self.sr_global)
        l_knee_rel = Point_Global_To_Local(np.array(l_knee_center), srt_l_thigh, self.sr_global)

        r_ankle_center = punto_medio(medias['r mall'], medias['r mall m'])
        l_ankle_center = punto_medio(medias['l mall'], medias['l mall m'])
        srt_r_shank = get_reference(medias['r mall'], r_knee_center, medias['r shank'])
        srt_l_shank = get_reference(medias['l mall'], l_knee_center, medias['l shank'])
        r_ankle_rel = Point_Global_To_Local(r_ankle_center, srt_r_shank, self.sr_global)
        l_ankle_rel = Point_Global_To_Local(l_ankle_center, srt_l_shank, self.sr_global)

        srt_r_ff_st = get_reference(medias['r met 1st'], medias['r met 5th'], medias['r foot'])
        srt_l_ff_st = get_reference(medias['l met 1st'], medias['l met 5th'], medias['l foot'])
        r_met_rel = Point_Global_To_Local(medias['r met'], srt_r_ff_st, self.sr_global)
        l_met_rel = Point_Global_To_Local(medias['l met'], srt_l_ff_st, self.sr_global)
        
        
        calib = {
            'medias': medias,
            'r_asis_local': r_asis_local,
            'l_asis_local': l_asis_local,
            'r_hip_rel': r_hip_center,
            'l_hip_rel': l_hip_center,
            'r_knee_rel': r_knee_rel,
            'l_knee_rel': l_knee_rel,
            'r_ankle_rel': r_ankle_rel,
            'l_ankle_rel': l_ankle_rel,
            'r_met_rel': r_met_rel,
            'l_met_rel': l_met_rel,
        }
        
        return calib
    
    def _cargar_y_filtrar_running(self):
        with Tdf(self.rutas['running']) as f:
                running = f.data3D

        mkrs_run = [
                'c7', 'r should', 'l should', 'sacrum', 'r tp', 'l tp', 'r thigh', 
                'r knee', 'r shank', 'r mall', 'r met 5th', 'r met 1st', 'r foot',
                'l thigh', 'l knee', 'l shank', 'l mall', 'l met 5th', 'l met 1st', 
                'l foot', 'r elb', 'r wri', 'l elb', 'l wri', 'r heel mk', 'l heel mk', 
                'r tal', 'l tal'
            ]
            
        signals = {}
        for i, m in enumerate(mkrs_run):
            self.progreso(0.2 + (i/len(mkrs_run))*0.3, f"Filtrando: {m}...")
            raw = running[m].data
            signals[m] = ft.suavizar_butterworth(ft.interpolar_datos(raw))
        
        return signals

    def _calcular_cinematica(self, signals, calib):
        """Calcula matrices de rotación y ángulos finales"""
        c7_rn, sacrum_rn = signals['c7'], signals['sacrum']
        r_tp_rn, l_tp_rn = signals['r tp'], signals['l tp']
        r_should_rn, l_should_rn = signals['r should'], signals['l should']
        r_knee_rn, l_knee_rn = signals['r knee'], signals['l knee']
        r_thigh_rn, l_thigh_rn = signals['r thigh'], signals['l thigh']
        r_mall_rn, l_mall_rn = signals['r mall'], signals['l mall']
        r_shank_rn, l_shank_rn = signals['r shank'], signals['l shank']
        r_met_1st_rn, r_met_5th_rn = signals['r met 1st'], signals['r met 5th']
        l_met_1st_rn, l_met_5th_rn = signals['l met 1st'], signals['l met 5th']
        r_foot_rn, l_foot_rn = signals['r foot'], signals['l foot']
        r_heel_rn, l_heel_rn = signals['r heel mk'], signals['l heel mk']
        r_tal_rn, l_tal_rn = signals['r tal'], signals['l tal']
        print("rheel",np.shape(r_heel_rn))

        #encuentro r y l asis usando el cluster de pelvis y las coordenadas locales de asis del standing
        sist_reference_triangle_rn = get_reference(r_tp_rn, l_tp_rn, sacrum_rn) 
        r_asis_rn = point_local_to_global(calib['r_asis_local'], sist_reference_triangle_rn, self.sr_global)
        l_asis_rn = point_local_to_global(calib['l_asis_local'], sist_reference_triangle_rn, self.sr_global)
        print("r_asis_rn", np.shape(r_asis_rn))
        #Armo sistema de referencia técnico de la pelvis
        p_orig_rn = punto_medio(r_asis_rn, l_asis_rn)
        cl_orig_rn = punto_medio(r_tp_rn, l_tp_rn)
        srt_pelvis_rn = get_reference(r_asis_rn, l_asis_rn, cl_orig_rn)
        print("srt_pelvis_rn", np.shape(srt_pelvis_rn))

        #traigo el centro de cadera en cada frame a coordenadas globales
        r_hip_rn = point_local_to_global(calib['r_hip_rel'], srt_pelvis_rn, self.sr_global)
        l_hip_rn = point_local_to_global(calib['l_hip_rel'], srt_pelvis_rn, self.sr_global)
        print("r_hip_rn", np.shape(r_hip_rn))

        # Sistema de referencia tácnico del tronco
        srt_trunk_rn = get_reference(r_should_rn, l_should_rn, c7_rn)

        #Sistema de referencia técnico del muslo
        srt_r_thigh_rn = get_reference(r_knee_rn, r_hip_rn, r_thigh_rn)
        srt_l_thigh_rn = get_reference(l_knee_rn, l_hip_rn, l_thigh_rn)

        #traigo el centro de rodilla en cada frame a coordenadas globales
        r_knee_center_rn = point_local_to_global(calib['r_knee_rel'], srt_r_thigh_rn, self.sr_global)
        l_knee_center_rn = point_local_to_global(calib['l_knee_rel'], srt_l_thigh_rn, self.sr_global)

        #Sistema de referencia técnico de la pierna
        srt_r_shank_rn = get_reference(r_mall_rn, r_knee_center_rn, r_shank_rn)
        srt_l_shank_rn = get_reference(l_mall_rn, l_knee_center_rn, l_shank_rn)

        #traigo el centro de tobillo en cada frame a coordenadas globales
        r_ankle_center_rn = point_local_to_global(calib['r_ankle_rel'], srt_r_shank_rn, self.sr_global)
        l_ankle_center_rn = point_local_to_global(calib['l_ankle_rel'], srt_l_shank_rn, self.sr_global)

        #Sistema de referencia tecnico de antepie
        srt_r_forefoot_rn = get_reference(r_met_1st_rn, r_met_5th_rn, r_foot_rn)
        srt_l_forefoot_rn = get_reference(l_met_1st_rn, l_met_5th_rn, l_foot_rn)

        #traigo met del standing
        r_met_rn = point_local_to_global(calib['r_met_rel'], srt_r_forefoot_rn, self.sr_global)
        l_met_rn = point_local_to_global(calib['l_met_rel'], srt_l_forefoot_rn, self.sr_global)

        #calculo punto medio entre 1st met y 5th met para tener referencia del 2do y 3er metatarsiano
        r_met_23_rn = punto_medio(r_met_1st_rn, r_met_5th_rn)
        l_met_23_rn = punto_medio(l_met_1st_rn, l_met_5th_rn)

        #armo sistemas de referencia anatomicos
        sr_run = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        sra_pelvis_rn = rotate_270_deg_z(srt_pelvis_rn)
        sra_pelvis_rn = rotate_270_deg_y(sra_pelvis_rn)
        sra_trunk_rn  = get_reference_form_vectors(cl_orig_rn, c7_rn, r_should_rn, l_should_rn)
        sra_trunk_rn  = rotate_270_deg_z(sra_trunk_rn)
        sra_trunk_rn  = rotate_270_deg_y(sra_trunk_rn)
        sra_r_thigh_rn = get_reference_anatomical(r_knee_center_rn, r_hip_rn, r_knee_rn, 0)
        sra_r_shank_rn = get_reference_anatomical(r_ankle_center_rn, r_knee_center_rn, r_mall_rn, 0)
        sra_r_shankunrot_rn = get_reference_anatomical(r_ankle_center_rn, r_knee_center_rn, r_knee_rn, 0)
        sra_l_thigh_rn = get_reference_anatomical(l_knee_center_rn, l_hip_rn, l_knee_rn, 1)
        sra_l_shank_rn = get_reference_anatomical(l_ankle_center_rn, l_knee_center_rn, l_mall_rn, 1)
        sra_l_shankunrot_rn = get_reference_anatomical(l_ankle_center_rn, l_knee_center_rn, l_knee_rn, 1)
        sra_l_forefoot_rn = get_reference_anatomical(l_met_23_rn, l_heel_rn, l_met_5th_rn, 1)
        sra_r_forefoot_rn = get_reference_anatomical(r_met_23_rn, r_heel_rn, r_met_5th_rn, 0)
        print("hasta aca SRA OK")

        sr_run_rn = expand_reference_matrix(sr_run, sra_pelvis_rn.shape[1])
        l_trunk_angles = get_euler_angles_rot(sr_run_rn, sra_trunk_rn)
        l_trunk_angles = np.array(change_sign_angles(l_trunk_angles,1,-1,1))
        r_trunk_angles = np.array(change_sign_angles(l_trunk_angles,-1,-1,1))
        #revisado contra mdx original. x transverso y frontal z sagital
        print("trunk angles ok")

        l_pelvis_angles = get_euler_angles_rot(sr_run_rn, sra_pelvis_rn)
        l_pelvis_angles = np.array(change_sign_angles(l_pelvis_angles,1,-1,1))
        r_pelvis_angles = np.array(change_sign_angles(l_pelvis_angles,-1,-1,1))
        #revisado contra mdx original. x transverso y frontal z sagital

        l_hip_angles = get_euler_angles_rot2(srt_pelvis_rn, sra_l_thigh_rn)
        r_hip_angles = get_euler_angles_rot2(srt_pelvis_rn, sra_r_thigh_rn)
        l_hip_angles = -1*l_hip_angles - [90,0,90]
        l_hip_angles[:, [1, 0, 2]] = np.column_stack((l_hip_angles[:, 0],
                                                    -l_hip_angles[:, 1],
                                                    l_hip_angles[:, 2]))
        r_hip_angles[:, [1, 0, 2]] = np.column_stack((r_hip_angles[:, 0] + 90,
                                                    -r_hip_angles[:, 1],
                                                    -r_hip_angles[:, 2] - 90))
        #revisado contra mdx original. x transverso y frontal z sagital
        print("hip angles OK")

        l_knee_angles = get_euler_angles_rot2(sra_l_shank_rn, sra_l_thigh_rn)
        r_knee_angles = get_euler_angles_rot2(sra_r_shank_rn, sra_r_thigh_rn)
        l_knee_angles[:, [0, 1, 2]] = np.column_stack((l_knee_angles[:, 0],
                                                    l_knee_angles[:, 1],
                                                    -l_knee_angles[:, 2]))
        r_knee_angles[:, [0, 1, 2]] = np.column_stack((-r_knee_angles[:, 0],
                                                    -r_knee_angles[:, 1],
                                                    -r_knee_angles[:, 2]))
        #revisado contra mdx original. x transverso y frontal z sagital
        print("kneee angles OK")


        l_ankle_angles = get_euler_angles_rot2(sra_l_shank_rn, sra_l_forefoot_rn)
        r_ankle_angles = get_euler_angles_rot2(sra_r_shank_rn, sra_r_forefoot_rn)
        l_ankle_angles[:, [0, 1, 2]] = np.column_stack((l_ankle_angles[:, 0],
                                                        -l_ankle_angles[:, 1],
                                                        -1*l_ankle_angles[:, 2]-90))
        r_ankle_angles[:, [0, 1, 2]] = np.column_stack((-1*r_ankle_angles[:, 0],
                                                        1*r_ankle_angles[:, 1],
                                                        -1*r_ankle_angles[:, 2]-90))
        #revisado contra mdx original. x no va es transverso de retro. y frontal (valgo retro) z sagital ok
        print("ankle angles OK")


        l_forefoot_angles = get_euler_angles_rot(sra_l_forefoot_rn, sr_run_rn)
        r_forefoot_angles = get_euler_angles_rot(sra_r_forefoot_rn, sr_run_rn)

        l_forefoot_angles[:, [0, 1, 2]] = np.column_stack((l_forefoot_angles[:, 0]+90,
                                                        -l_forefoot_angles[:, 1],
                                                        l_forefoot_angles[:, 2]))
        r_forefoot_angles[:, [0, 1, 2]] = np.column_stack((r_forefoot_angles[:, 0]+90,
                                                        r_forefoot_angles[:, 1],
                                                        -r_forefoot_angles[:, 2]))
        print("forefoot angles OK")

        #fl.save_euler_angles_to_csv(r_forefoot_angles, "data/r_ff_angles.csv", include_frame=True)
        #fl.save_euler_angles_to_csv(l_forefoot_angles, "data/l_ff_angles.csv", include_frame=True)


        #foot progresion and foot vs floor angle (used for type of initial contact (-)forefoot/(+)Heel ))
        vector_3D = two_point_unit_vector(l_heel_rn, l_met_23_rn)
        l_foot_progression_angle  = angulo_con_plano_sagital(vector_3D, 1)
        l_foot_floor_angle = angle_between_2_unit_vectors(vector_3D, sr_run_rn[1]) - 90
        vector_3D = two_point_unit_vector(r_heel_rn, r_met_23_rn)
        r_foot_progression_angle = angulo_con_plano_sagital(vector_3D, 0)
        r_foot_floor_angle = angle_between_2_unit_vectors(vector_3D, sr_run_rn[1]) - 90

        #cargo todo en un vector 3D x: progresion del pie y y: angulo del pie con el piso contacto inicial
        l_footabs_angles = np.column_stack([
            l_foot_progression_angle,
            l_foot_floor_angle,
            l_foot_floor_angle
        ])

        r_footabs_angles = np.column_stack([
            r_foot_progression_angle,
            r_foot_floor_angle,
            r_foot_floor_angle
        ])
        print("foot angles OK")


        #rearfoot valgus
        l_vector_rear = two_point_unit_vector(l_heel_rn, l_tal_rn )
        l_vec_shank_vert = two_point_unit_vector(l_ankle_center_rn, l_knee_center_rn)
        l_vec_shank_hor = two_point_unit_vector(l_ankle_center_rn, l_mall_rn)
        l_rearfoot_absolute = calcular_angulo_proyectado(l_vector_rear, l_vec_shank_hor, l_vec_shank_vert)
        r_vector_rear = two_point_unit_vector(r_heel_rn, r_tal_rn )
        r_vec_shank_vert = two_point_unit_vector(r_ankle_center_rn, r_knee_center_rn)
        r_vec_shank_hor = two_point_unit_vector(r_mall_rn, r_ankle_center_rn)
        r_rearfoot_absolute = calcular_angulo_proyectado(r_vector_rear, r_vec_shank_hor, r_vec_shank_vert)
        print("rearfoot angles OK")
        nframes = l_rearfoot_absolute.shape[0]  

        #angulo de la tibia
        l_tibia_absolute = calcular_angulo_proyectado(l_vec_shank_vert, sr_run_rn[2], sr_run_rn[1])
        r_tibia_absolute = calcular_angulo_proyectado(r_vec_shank_vert, sr_run_rn[2], sr_run_rn[1])
        l_tibia_absolute = 90 - l_tibia_absolute
        r_tibia_absolute = 90 - r_tibia_absolute
        print("tibia angles OK")

        self.angulos = {
            'l': {
                'trunk': l_trunk_angles,       # shape = (n_frames, 3)
                'pelvis': l_pelvis_angles,
                'hip': l_hip_angles,
                'knee': l_knee_angles,
                'ankle': l_ankle_angles,
                'forefoot': l_forefoot_angles,
                'footabs': l_footabs_angles,
                'shank': l_tibia_absolute
            },
            'r': {
                'trunk': r_trunk_angles,       # shape = (n_frames, 3)
                'pelvis': r_pelvis_angles,
                'hip': r_hip_angles,
                'knee': r_knee_angles,
                'ankle': r_ankle_angles,
                'forefoot': r_forefoot_angles,
                'footabs': r_footabs_angles,
                'shank': r_tibia_absolute
            }
        }
        print("diccionario angles OK")

    pass