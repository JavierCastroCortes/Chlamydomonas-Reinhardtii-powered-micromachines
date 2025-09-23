import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

def save_general_simulation_data(vels_sistem_gral, tray_sistem_gral, dt=1/240, 
                                metadata=None, filename="simulaciones_generales.json"):
    """
    Guarda datos de múltiples simulaciones y calcula estadísticas de tendencia central
    
    Args:
        vels_sistem_gral: Lista de listas con velocidades de cada simulación
        tray_sistem_gral: Lista de listas con trayectorias de cada simulación
        dt: Paso de tiempo utilizado en las simulaciones
        metadata: Metadatos adicionales (opcional)
        filename: Nombre del archivo de salida
    """
    
    # Verificar que tenemos el mismo número de simulaciones para velocidades y trayectorias
    assert len(vels_sistem_gral) == len(tray_sistem_gral), \
        "El número de simulaciones debe ser igual para velocidades y trayectorias"
    
    n_simulaciones = len(vels_sistem_gral)
    
    # Crear estructura de datos principal
    datos_generales = {
        "metadata": {
            "fecha_creacion": datetime.now().isoformat(),
            "numero_simulaciones": n_simulaciones,
            "dt_segundos": dt,
            "frecuencia_muestreo_hz": 1/dt,
            **({} if metadata is None else metadata)
        },
        "simulaciones": [],
        "estadisticas": {}
    }
    
    # Procesar cada simulación individual
    for i, (vels, tray) in enumerate(zip(vels_sistem_gral, tray_sistem_gral)):
        tiempos = [j * dt for j in range(len(vels))]
        
        simulacion_data = {
            "id_simulacion": i,
            "duracion_segundos": len(vels) * dt,
            "muestras_velocidad": len(vels),
            "muestras_trayectoria": len(tray),
            "datos": [
                {
                    "tiempo_s": round(t, 6),
                    "velocidad_μm_s": round(vel, 6),
                    "posicion_m": [round(x, 8), round(y, 8), round(z, 8)]
                }
                for t, vel, (x, y, z) in zip(tiempos, vels, tray)
            ]
        }
        
        datos_generales["simulaciones"].append(simulacion_data)
    
    # Calcular estadísticas de tendencia central para las velocidades
    calcular_estadisticas_velocidad(datos_generales, vels_sistem_gral, dt)
    
    # Guardar el archivo
    with open(filename, 'w') as f:
        json.dump(datos_generales, f, indent=2)
    
    print(f"Datos de {n_simulaciones} simulaciones guardados en {filename}")
    return datos_generales

def calcular_estadisticas_velocidad(datos_generales, vels_sistem_gral, dt):
    """Calcula estadísticas de tendencia central para las velocidades"""
    
    # Encontrar la longitud mínima entre todas las simulaciones
    min_length = min(len(vels) for vels in vels_sistem_gral)
    
    # Crear array con las velocidades de todas las simulaciones (truncadas a la longitud mínima)
    todas_velocidades = np.array([vels[:min_length] for vels in vels_sistem_gral])
    
    # Calcular estadísticas por tiempo
    tiempos = np.arange(min_length) * dt
    medias = np.mean(todas_velocidades, axis=0)
    medianas = np.median(todas_velocidades, axis=0)
    desviaciones = np.std(todas_velocidades, axis=0)
    minimos = np.min(todas_velocidades, axis=0)
    maximos = np.max(todas_velocidades, axis=0)
    
    # Calcular estadísticas globales
    todas_velocidades_flat = todas_velocidades.flatten()
    
    datos_generales["estadisticas"] = {
        "por_tiempo": [
            {
                "tiempo_s": round(t, 6),
                "media_velocidad": round(media, 6),
                "mediana_velocidad": round(mediana, 6),
                "desviacion_estandar": round(desv, 6),
                "minimo_velocidad": round(minimo, 6),
                "maximo_velocidad": round(maximo, 6)
            }
            for t, media, mediana, desv, minimo, maximo in zip(
                tiempos, medias, medianas, desviaciones, minimos, maximos
            )
        ],
        "globales": {
            "media_global": round(float(np.mean(todas_velocidades_flat)), 6),
            "mediana_global": round(float(np.median(todas_velocidades_flat)), 6),
            "desviacion_global": round(float(np.std(todas_velocidades_flat)), 6),
            "minimo_global": round(float(np.min(todas_velocidades_flat)), 6),
            "maximo_global": round(float(np.max(todas_velocidades_flat)), 6),
            "rango_global": round(float(np.ptp(todas_velocidades_flat)), 6)
        }
    }

# Función para cargar y visualizar los datos guardados
def cargar_y_visualizar_estadisticas(filename):
    """Carga los datos guardados y muestra resumen de estadísticas"""
    
    with open(filename, 'r') as f:
        datos = json.load(f)
    
    print(f"Resumen de {datos['metadata']['numero_simulaciones']} simulaciones")
    print(f"Duración máxima: {max(sim['duracion_segundos'] for sim in datos['simulaciones']):.2f}s")
    
    stats = datos['estadisticas']['globales']
    print("\nEstadísticas globales de velocidad:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return datos