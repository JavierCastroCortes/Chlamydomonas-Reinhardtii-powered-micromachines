import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

def calcular_mse_trayectorias(carpeta_con_carga, carpeta_sin_carga, num_micromaquinas=5):
    """
    Calcula el MSE entre trayectorias de micromáquinas con y sin carga
    
    Args:
        carpeta_con_carga: Ruta a la carpeta con archivos de carga
        carpeta_sin_carga: Ruta a la carpeta con archivos sin carga
        num_micromaquinas: Número de micromáquinas a procesar
    """
    
    mse_values = []
    
    for i in range(1, num_micromaquinas + 1):
        # Cargar archivos
        archivo_con_carga = os.path.join(carpeta_con_carga, f"simulacion_promediada_M{i}.json")
        archivo_sin_carga = os.path.join(carpeta_sin_carga, f"simulacion_promediada_M{i}.json")
        
        try:
            with open(archivo_con_carga, 'r') as f:
                datos_con_carga = json.load(f)
            
            with open(archivo_sin_carga, 'r') as f:
                datos_sin_carga = json.load(f)
                
            #eextraemos tiempos y posiciones
            tiempos_con = [punto['tiempo_s'] for punto in datos_con_carga['datos']]
            posiciones_con = [punto['posicion_m'] for punto in datos_con_carga['datos']]
            
            tiempos_sin = [punto['tiempo_s'] for punto in datos_sin_carga['datos']]
            posiciones_sin = [punto['posicion_m'] for punto in datos_sin_carga['datos']]
            
            #converimos a arrays numpy
            tiempos_con = np.array(tiempos_con)
            posiciones_con = np.array(posiciones_con)
            
            tiempos_sin = np.array(tiempos_sin)
            posiciones_sin = np.array(posiciones_sin)
            
            #interpolamooos para tener los mismos puntos de tiempo
            #usaaaaamos la trayectoria sin carga como referencia
            posiciones_con_interp = np.zeros_like(posiciones_sin)
            
            for dim in range(3):  # Para las tres dimensiones (x, y, z)
                interp_func = interp1d(tiempos_con, posiciones_con[:, dim], 
                                      bounds_error=False, fill_value="extrapolate")
                posiciones_con_interp[:, dim] = interp_func(tiempos_sin)
            
            #calculamosssss MSE para cada dimensión
            mse_x = np.mean((posiciones_con_interp[:, 0] - posiciones_sin[:, 0]) ** 2)
            mse_y = np.mean((posiciones_con_interp[:, 1] - posiciones_sin[:, 1]) ** 2)
            mse_z = np.mean((posiciones_con_interp[:, 2] - posiciones_sin[:, 2]) ** 2)
            
            # MSE total (promedio de x,y y z)
            mse_total = (mse_x + mse_y + mse_z) / 3
            
            mse_values.append({
                'micromaquina': i,
                'mse_total': mse_total,
                'mse_x': mse_x,
                'mse_y': mse_y,
                'mse_z': mse_z
            })
            
        except FileNotFoundError:
            print(f"Advertencia: No se encontraron archivos para la micromáquina {i}")
            continue
    
    return mse_values

def graficar_mse(mse_values):
    """
    Generamos los gráficos de los valores MSE
    """
    if not mse_values:
        print("No hay datos para graficar")
        return
    
    micromaquinas = [m['micromaquina'] for m in mse_values]
    mse_totales = [m['mse_total'] for m in mse_values]
    mse_x = [m['mse_x'] for m in mse_values]
    mse_y = [m['mse_y'] for m in mse_values]
    mse_z = [m['mse_z'] for m in mse_values]
    
    #grafico de MSE total por micromáquina
    plt.figure(figsize=(12, 6))
    plt.bar(micromaquinas, mse_totales, alpha=0.7)
    plt.xlabel('Micromáquina')
    plt.ylabel('MSE Total')
    plt.title('Error Cuadrático Medio (MSE) entre Trayectorias Con y Sin Carga')
    plt.grid(True, alpha=0.3)
    plt.savefig('mse_total.png')
    plt.show()
    
    #gráfico comparativo por dimensiones
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(micromaquinas))
    width = 0.2
    
    plt.bar(x_pos - width, mse_x, width, label='MSE X', alpha=0.7)
    plt.bar(x_pos, mse_y, width, label='MSE Y', alpha=0.7)
    plt.bar(x_pos + width, mse_z, width, label='MSE Z', alpha=0.7)
    
    plt.xlabel('Micromáquina')
    plt.ylabel('MSE')
    plt.title('MSE por Dimensión y Micromáquina')
    plt.xticks(x_pos, micromaquinas)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mse_por_dimension.png')
    plt.show()
    
    #grafico de MSE total con línea de tendencia
    plt.figure(figsize=(12, 6))
    plt.plot(micromaquinas, mse_totales, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Micromáquina')
    plt.ylabel('MSE Total')
    plt.title('Evolución del MSE Total por Micromáquina')
    plt.grid(True, alpha=0.3)
    plt.savefig('ResultadosMSE/mse_evolucion.png')
    plt.show()

if __name__ == "__main__":
    # Calcular MSE
    mse_resultados = calcular_mse_trayectorias("ResultadosMSE/ConCarga", "ResultadosMSE/SinCarga")
    
    # Mostrar resultados en consola
    print("Resultados MSE:")
    for resultado in mse_resultados:
        print(f"Micromáquina {resultado['micromaquina']}:")
        print(f"  MSE Total: {resultado['mse_total']:.6e}")
        print(f"  MSE X: {resultado['mse_x']:.6e}")
        print(f"  MSE Y: {resultado['mse_y']:.6e}")
        print(f"  MSE Z: {resultado['mse_z']:.6e}")
        print()
    
    
    graficar_mse(mse_resultados)
    
    #guardamos resultados en archivo
    with open('ResultadosMSE/resultados_mse.json', 'w') as f:
        json.dump(mse_resultados, f, indent=2)