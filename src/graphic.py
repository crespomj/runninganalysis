import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def graficar_curvas_tk(contenedor_tk, data_sujeto_cond, modo="ambos", n_puntos=100):
    """
    Incrusta los gráficos de biomecánica en un contenedor de Tkinter.
    """
    segmentos = ["trunk", "pelvis", "hip", "knee", "ankle", "forefoot"]
    planos = ["x", "y", "z"]
    
    # 1. Creamos la figura (sin plt.show())
    fig, axes = plt.subplots(len(segmentos), 3, figsize=(15, 30), sharex=True)
    
    for i, seg in enumerate(segmentos):
        for j, plano in enumerate(planos):
            ax = axes[i, j]
            
            # Caso especial shank (o validación de segmentos)
            if seg == "shank" and plano != "x":
                ax.axis("off")
                continue
            
            # Extraer datos de forma segura
            lado_l = data_sujeto_cond.get("l", {}).get(seg, {}).get(plano)
            lado_r = data_sujeto_cond.get("r", {}).get(seg, {}).get(plano)
            
            if lado_l is None or lado_r is None:
                ax.axis("off")
                continue
            
            x_vals = np.linspace(0, 100, n_puntos + 1)
            
            # --- Lado Izquierdo (Rojo) ---
            if modo in ["ciclos", "ambos"]:
                for ciclo in lado_l["ciclos_norm"]:
                    ax.plot(x_vals, ciclo, color="red", alpha=0.3)
            if modo in ["promedio", "ambos"]:
                mean_l = np.mean(lado_l["ciclos_norm"], axis=0)
                ax.plot(x_vals, mean_l, color="red", linewidth=2, label="Esq") # 'Esq' es 'Izq' en portugués

            # --- Lado Derecho (Verde) ---
            if modo in ["ciclos", "ambos"]:
                for ciclo in lado_r["ciclos_norm"]:
                    ax.plot(x_vals, ciclo, color="darkgreen", alpha=0.3)
            if modo in ["promedio", "ambos"]:
                mean_r = np.mean(lado_r["ciclos_norm"], axis=0)
                ax.plot(x_vals, mean_r, color="darkgreen", linewidth=2, label="Dir") # 'Dir' es 'Der' en portugués
            
            # Estética (Agregamos títulos bilingües para practicar)
            ax.set_title(f"{seg.upper()} - Plano {plano.upper()}")
            ax.grid(True, alpha=0.3)
            if i == len(segmentos)-1:
                ax.set_xlabel("% ciclo")

    fig.tight_layout()

    # 2. Integramos la figura en el Canvas de Tkinter
    canvas = FigureCanvasTkAgg(fig, master=contenedor_tk)
    canvas.draw()
    
    # 3. Empaquetamos el widget en la interfaz
    widget = canvas.get_tk_widget()
    widget.pack(side="top", fill="both", expand=True)
    
    return canvas # Devolvemos el objeto para poder destruirlo/limpiarlo luego


def graficar_curvas(data_sujeto_cond, modo="ambos", n_puntos=100):
    """
    Grafica las curvas de todos los segmentos/articulaciones organizadas en un grid 8x3.
    
    Parámetros:
    - data_sujeto_cond: dict = data[sujeto][condicion]
    - modo: "promedio", "ciclos" o "ambos"
    - n_puntos: cantidad de puntos normalizados (por defecto 100 → 101 filas)
    """
    
    segmentos = ["trunk", "pelvis", "hip", "knee", "ankle", "forefoot"]
    planos = ["x", "y", "z"]
    
    fig, axes = plt.subplots(len(segmentos), 3, figsize=(15, 30), sharex=True)
    
    for i, seg in enumerate(segmentos):
        for j, plano in enumerate(planos):
            
            # Para shank solo graficamos el plano X
            if seg == "shank" and plano != "x":
                axes[i, j].axis("off")
                continue
            
            ax = axes[i, j]
            
            # Extraer datos izquierdo/derecho
            curvas_l = data_sujeto_cond["l"][seg].get(plano, None)
            curvas_r = data_sujeto_cond["r"][seg].get(plano, None)
            
            if curvas_l is None or curvas_r is None:
                ax.axis("off")
                continue
            
            x_vals = np.linspace(0, 100, n_puntos+1)
            
            # Lado izquierdo
            if modo in ["ciclos", "ambos"]:
                for ciclo in curvas_l["ciclos_norm"]:
                    ax.plot(x_vals, ciclo, color="red", alpha=0.3)
            if modo in ["promedio", "ambos"]:
                mean_curve = np.mean(curvas_l["ciclos_norm"], axis=0)
                ax.plot(x_vals, mean_curve, color="red", linewidth=2, label="Izq")
            
            # Lado derecho
            if modo in ["ciclos", "ambos"]:
                for ciclo in curvas_r["ciclos_norm"]:
                    ax.plot(x_vals, ciclo, color="darkgreen", alpha=0.3)
            if modo in ["promedio", "ambos"]:
                mean_curve = np.mean(curvas_r["ciclos_norm"], axis=0)
                ax.plot(x_vals, mean_curve, color="darkgreen", linewidth=2, label="Der")
            
            # Estética
            ax.set_title(f"{seg} - {plano}")
            ax.grid(True, alpha=0.3)
            if i == len(segmentos)-1:
                ax.set_xlabel("% ciclo")
    
    plt.tight_layout()
    plt.show()
