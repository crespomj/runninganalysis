import numpy as np
import matplotlib.pyplot as plt

def plot_runner_profile(params, values, profile_name="Perfil del Corredor"):
    """
    Crea un gráfico de araña para perfiles biomecánicos.
    
    Args:
        params (list): Nombres de las 9 variables.
        values (list): Valores estandarizados (Z-scores).
        profile_name (str): Título del gráfico.
    """
    # Número de variables
    num_vars = len(params)

    # Calcular ángulos para cada eje
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # El gráfico debe ser circular, cerramos el loop
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Dibujar los ejes y etiquetas
    plt.xticks(angles[:-1], params, color='grey', size=10)

    # Dibujar los niveles (Z-scores de -3 a 3 como ejemplo)
    ax.set_rlabel_position(0)
    plt.yticks([-2, 0, 2], ["-2", "0", "+2"], color="grey", size=8)
    plt.ylim(-3, 3)

    # Graficar los datos
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=profile_name)
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(profile_name, size=15, color='blue', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.show()

# Ejemplo de uso:
variables = [
    'Cadencia', 'T. Apoyo', 'Osc. Vertical', 'Duty Factor', 
    'Rigidez Vert.', 'Peak GRF', 'Fluidez AP', 'Fluidez Lat', 'Fluidez Vert'
]
# Valores ejemplo (Z-scores de un perfil hipotético)
z_scores = [1.2, -0.5, 2.1, -1.1, 0.4, 1.8, -0.2, 0.1, 0.5]

plot_runner_profile(variables, z_scores, "Perfil P4 (Alta Carga)")