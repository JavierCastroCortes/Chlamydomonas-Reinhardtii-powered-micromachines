import json
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_from_json(simulation, m,filename):
    # Cargar el archivo JSON
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extraer datos
    tiempos = []
    posiciones_x = []
    posiciones_y = [] 
    posiciones_z = []
    velocidades = []
    
    for punto in data['datos']:
        tiempos.append(punto['tiempo_s'])
        posiciones_x.append(punto['posicion_m'][0])
        posiciones_y.append(punto['posicion_m'][1])
        posiciones_z.append(punto['posicion_m'][2])
        velocidades.append(punto['velocidad_\u03bcm_s'])
    
    # Convertir a arrays de numpy
    tiempos = np.array(tiempos)
    posiciones_x = np.array(posiciones_x)
    posiciones_y = np.array(posiciones_y)
    posiciones_z = np.array(posiciones_z)
    velocidades = np.array(velocidades)
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Gráfico 3D de la trayectoria
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(posiciones_x, posiciones_y, posiciones_z, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(posiciones_x[0], posiciones_y[0], posiciones_z[0], c='green', s=100, label='Inicio')
    ax1.scatter(posiciones_x[-1], posiciones_y[-1], posiciones_z[-1], c='red', s=100, label='Fin')
    ax1.set_xlabel('Position X (m)')
    ax1.set_ylabel('Position Y (m)')
    ax1.set_zlabel('Position Z (m)')
    ax1.set_title('3D Trajectory of the Micromachine')
    ax1.legend()
    ax1.grid(True)
    
    # Proyecciones 2D
    ax2 = fig.add_subplot(232)
    ax2.plot(posiciones_x, posiciones_y, 'r-', linewidth=1)
    ax2.scatter(posiciones_x[0], posiciones_y[0], c='green', s=50, label='Inicio')
    ax2.scatter(posiciones_x[-1], posiciones_y[-1], c='red', s=50, label='Fin')
    ax2.set_xlabel('Position X (m)')
    ax2.set_ylabel('Position Y (m)')
    ax2.set_title('Projection XY')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(233)
    ax3.plot(posiciones_x, posiciones_z, 'g-', linewidth=1)
    ax3.scatter(posiciones_x[0], posiciones_z[0], c='green', s=50, label='Inicio')
    ax3.scatter(posiciones_x[-1], posiciones_z[-1], c='red', s=50, label='Fin')
    ax3.set_xlabel('Position X (m)')
    ax3.set_ylabel('Position Z (m)')
    ax3.set_title('Projection XZ')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = fig.add_subplot(234)
    ax4.plot(posiciones_y, posiciones_z, 'm-', linewidth=1)
    ax4.scatter(posiciones_y[0], posiciones_z[0], c='green', s=50, label='Inicio')
    ax4.scatter(posiciones_y[-1], posiciones_z[-1], c='red', s=50, label='Fin')
    ax4.set_xlabel('Position Y (m)')
    ax4.set_ylabel('Position Z (m)')
    ax4.set_title('Projection YZ')
    ax4.legend()
    ax4.grid(True)
    
    # Velocidad vs tiempo
    ax5 = fig.add_subplot(235)
    ax5.plot(tiempos, velocidades, 'c-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (µm/s)')
    ax5.set_title('Speed ​​vs Time')
    ax5.grid(True)
    
    # Posición vs tiempo
    ax6 = fig.add_subplot(236)
    ax6.plot(tiempos, posiciones_x, 'r-', label='X', linewidth=1)
    ax6.plot(tiempos, posiciones_y, 'g-', label='Y', linewidth=1)
    ax6.plot(tiempos, posiciones_z, 'b-', label='Z', linewidth=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('Position vs Time')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'M{m}/'+str(simulation)+f'.png')
    #plt.show()
    
    # Información adicional
    print(f"Duración total: {data['metadata']['duracion_total_segundos']} s")
    print(f"Frecuencia de muestreo: {data['metadata']['frecuencia_muestreo_hz']} Hz")
    print(f"Número de puntos: {len(tiempos)}")
    print(f"Desplazamiento total X: {posiciones_x[-1] - posiciones_x[0]:.2e} m")
    print(f"Desplazamiento total Y: {posiciones_y[-1] - posiciones_y[0]:.2e} m")
    print(f"Desplazamiento total Z: {posiciones_z[-1] - posiciones_z[0]:.2e} m")
