import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import math
from trayectoria_micromaquina import plot_trajectory_from_json
from save_general_simulation_data import save_general_simulation_data


PATH_GRAL = "/Users/javimalikian/Documents/Tesis Master/Tesis Master/Algoritmos/PeerJ code/"
VISCOSIDAD_AGUA = 8.9e-4  # Pa·s (viscosidad a 25°C)
RADIO_CARACTERISTICO = 7e-6  # 5 μm

voxels_hist_general = []
fitness_hist = []

mejor_fitness = float('inf')
generaciones_sin_mejora = 0
m = 0
generacion=0
n_microalgas = 0

# Parámetros de los voxels
masa = 15e-10
soporte_c = [0.5, 0.5, 0.5, 1]
alga_c = [0, 1, 0, 1]
cubo_size = 0.5

# Parámetros fuerzade la microalga
alga_force = 25e-12  # Fuerza en Newtons
alga_force_ex = 20e-12  # Componente extra para aleatoriedad

#parametros de contaminantes para fuerza de arrastre
COEFICIENTE_ARRASTRE_CONTAMINANTES = 1e-26  # k, ajustable
DENSIDAD_CONTAMINANTES_BASE = 1e12  # partículas/m³, ajustable

#parámetros para el modelo de captura
#DENSIDAD_INICIAL_CONTAMINANTES = 1e15  # partículas/m³
DENSIDAD_INICIAL_CONTAMINANTES = 11340  # partículas/m³
EFICIENCIA_CAPTURA = 0.1  # Probabilidad de capturar una partícula
AREA_EFECTIVA_CAPTURA = (10e-7)**2  # Área efectiva de captura (m²)
MASA_POR_PARTICULA = 2.4e-20  # kg/partícula

#diccionario para rastrear la densidad por región (simula un campo espacial)
densidad_contaminantes_regiones = {}
TAMANO_REGION = 0.1  # metros (tamaño de las regiones espaciales)

masa_micromaquina = 120e-11  # masa inicial de la micromáquina (kg)


def load_micromachine_new_format(filename):
        """
        Carga el cromosoma en el nuevo formato JSON EJEMPLO
        {
          "id": "0",
          "voxels": [
            {"p": [0,0,0], "c": 0, "f": [1,-1,0]},
            {"p": [1,0,0], "c": 1},
            ...
          ]
        }
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Archivo no encontrado: {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extraer la lista de voxels
        voxels_parametros = data["voxels"]
        chromosome_id = data.get("id", "unknown")
        
        print(f"Cargado micromaquina ID: {chromosome_id} con {len(voxels_parametros)} voxels")
        return voxels_parametros, chromosome_id

def crete_voxel(masa=120e-10, tamano=0.5, posicion=[0,0,0], color=1):
        color_c = [0.5, 0.5, 0.5, 1] if color == 1 else [0, 1, 0, 1]
        global n_microalgas
        
        if color == 0:
            n_microalgas = n_microalgas + 1
        
        # guardamos las posiciones de los cubos en un arreglo
        #config_cubos[posicion[0], posicion[1], posicion[2]] = 1
        
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tamano, tamano, tamano])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tamano, tamano, tamano], rgbaColor=color_c)
        body = p.createMultiBody(baseMass=masa, baseCollisionShapeIndex=shape, 
                                baseVisualShapeIndex=visual, basePosition=posicion)
        voxel_bodies.append(body)
        return body

def crea_constraints():
        voxel_size = 1.0  # Ajusta según la separación entre centros de voxels
        for i, voxel_i in enumerate(voxels_parametros):
            pos_i = np.array(voxel_i["p"])
            for j, voxel_j in enumerate(voxels_parametros[i+1:], start=i+1):  # Evita duplicados
                pos_j = np.array(voxel_j["p"])
                distance = np.linalg.norm(pos_i - pos_j)
                
                # Verifica si son adyacentes (distancia = 1 en al menos un eje)
                if distance == 1.0:
                    # Punto medio entre los dos voxels (en coordenadas locales)
                    anchor = (np.array(pos_j) - np.array(pos_i)) * voxel_size / 2
                    
                    p.createConstraint(
                        parentBodyUniqueId=voxel_bodies[i],
                        parentLinkIndex=-1,
                        childBodyUniqueId=voxel_bodies[j],
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=anchor,
                        childFramePosition=-anchor
                    )

def aplicar_fuerzas_alga():
        for idx, voxel in enumerate(voxels_parametros):
            if voxel["c"] == 0 and "f" in voxel:
                # Uso del vector de fuerza del JSON
                
                random_force_1 = random.choices([alga_force_ex, -alga_force_ex, 0], weights=[0.4, 0.4, 0.2], k=1)[0]
                random_force_2 = random.choices([alga_force_ex, -alga_force_ex, 0], weights=[0.4, 0.4, 0.2], k=1)[0]
                
                #random_force_1 = random.choices([alga_force_ex,  0], weights=[0.8, 0.2], k=1)[0]
                #random_force_2 = random.choices([alga_force_ex,  0], weights=[0.8, 0.2], k=1)[0]
                
                force_vector = np.array(voxel["f"]) * alga_force
                #generar un vector de fuerza basado en la dirección especificada en el JSON
                if voxel["f"][0] == 1 or voxel["f"][0] == -1:
                    force_vector = np.array([alga_force, random_force_1, random_force_2])
                elif voxel["f"][1] == 1 or voxel["f"][1] == -1:
                    force_vector = np.array([random_force_1, alga_force, random_force_2])
                elif voxel["f"][2] == 1 or voxel["f"][2] == -1:
                    force_vector = np.array([random_force_1, random_force_2, alga_force])
                #print(f"Aplicando fuerza {force_vector} al voxel {idx} en posición {voxel['p']}")
                        
                # Aplicar fuerza según el vector especificado en el JSON
                p.applyExternalForce(voxel_bodies[idx], -1, forceObj=force_vector.tolist(), posObj=[0, 0, 0], flags=p.LINK_FRAME)

# Obtener la velocidad lineal y convertir a μm/s
def obt_vel_lineal(vel_lin):
        vx, vy, vz = vel_lin
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        speed *= 1e6  # Convertir a μm/s
        return speed

# Calcular error cuadrático medio y distancia al objetivo
def calc_error_cuad_medio(posiciones_simuladas):
        ruta_trayectoria = PATH_GRAL+"Pybullet/trayectoria_lineal.json"
        try:
            with open(ruta_trayectoria) as f:
                data = json.load(f)
            # Extraer la trayectoria objetivo y convertir a metros
            trayectoria_objetivo = [tuple(np.array(p["posicion_um"]) * 1e-6) for p in data["trayectoria"]]

            # Truncar ambas listas a la misma longitud
            N = min(len(posiciones_simuladas), len(trayectoria_objetivo))
            sim = np.array(posiciones_simuladas[:N])
            ref = np.array(trayectoria_objetivo[:N])

            # Calcular error cuadrático medio
            ecm = np.mean(np.sum((sim - ref)**2, axis=1))
            
            # Calcular distancia al punto objetivo final
            ultimo_objetivo_um = data["trayectoria"][-1]["posicion_um"]
            ultimo_objetivo_m = np.array(ultimo_objetivo_um) * 1e-6
            ultimo_simulado_m = np.array(posiciones_simuladas[-1])
            distancia_m = np.linalg.norm(ultimo_objetivo_m - ultimo_simulado_m)
            distancia_um = distancia_m * 1e6

            return ecm, distancia_um
            
        except FileNotFoundError:
            print(f"Archivo de trayectoria no encontrado: {ruta_trayectoria}")
            return float('inf'), float('inf')

# Obtener la velocidad promedio de la micromáquina  
def get_micromachine_velocity(voxel_bodies):
        velocities = []
        for voxel in voxel_bodies:
            vel, _ = p.getBaseVelocity(voxel)
            velocities.append(np.array(vel))
        # promedio
        avg_velocity = np.mean(velocities, axis=0)
        return avg_velocity

# Guardar datos con timestamps 
def save_data_time(m,vels_sistem, tray_sistem, dt=1/240, filename="simulacion_con_tiempo.json"):
        # Crear lista de tiempos
        '''
         vels_sistem, 
            tray_sistem, 
            duracion_total=duracion,
            filename=f"simulacion_{chromosome_id}_con_tiempo.json")
        '''
        tiempos = [i * dt for i in range(len(vels_sistem))]
        datos = {
            "metadata": {
                "micromasquina": m,
                "frecuencia_muestreo_hz": 1/dt,
                "dt_segundos": dt,
                "duracion_total_segundos": len(vels_sistem) * dt
            },
            "datos": [
                {
                    "tiempo_s": round(t, 6),
                    "velocidad_μm_s": round(vel, 6),
                    "posicion_m": [round(x, 8), round(y, 8), round(z, 8)]
                }
                for t, vel, (x, y, z) in zip(tiempos, vels_sistem, tray_sistem)
            ]
        }
        
        # Guardar en chunks para evitar problemas de memoria
        with open(filename, 'w') as f:
            json.dump(datos, f, indent=2)
        print(f"Datos con timestamp guardados en {filename}")

#aplicar fuerza de arrastrea cada uno de los voxels
def aplicar_fuerza_arrastre_sin_contaminantes():
    """Aplica fuerza de arrastre al centro de masa de toda la micromáquina"""
    # Calcular velocidad del centro de masa
    vel_total = np.zeros(3)
    for body in voxel_bodies:
        vel, _ = p.getBaseVelocity(body)
        vel_total += np.array(vel)
        vel_promedio = vel_total / len(voxel_bodies)
    
    # Fuerza de arrastre para toda la estructura
    fuerza_arrastre_total = -6 * math.pi * VISCOSIDAD_AGUA * RADIO_CARACTERISTICO * vel_promedio
    
    # Distribuir fuerza por masa relativa
    for body in voxel_bodies:
        p.applyExternalForce(body, -1, 
                           forceObj=fuerza_arrastre_total.tolist(), 
                           posObj=[0, 0, 0], 
                           flags=p.LINK_FRAME)

def aplicar_fuerza_arrastre_con_contaminantes_anterior():
    """Aplica fuerza de arrastre al centro de masa de toda la micromáquina, incluyendo contaminantes"""
    # Calcular velocidad del centro de masa
    vel_total = np.zeros(3)
    #pos_total = np.zeros(3)
    for body in voxel_bodies:
        vel, _ = p.getBaseVelocity(body)
        vel_total += np.array(vel)
        vel_promedio = vel_total / len(voxel_bodies)
    #pos_promedio = pos_total / len(voxel_bodies)  # centro de masa

    # Fuerza de arrastre hidrodinámica
    fuerza_arrastre_hidro = -6 * math.pi * VISCOSIDAD_AGUA * RADIO_CARACTERISTICO * vel_promedio

    # Fuerza de arrastre debido a contaminantes (densidad constante)
    fuerza_arrastre_contaminantes = -COEFICIENTE_ARRASTRE_CONTAMINANTES * DENSIDAD_CONTAMINANTES_BASE * vel_promedio

    # Fuerza total de arrastre
    fuerza_arrastre_total = fuerza_arrastre_hidro + fuerza_arrastre_contaminantes

    # Distribuir fuerza por masa relativa? En realidad, estamos aplicando la misma fuerza a cada voxel.
    # Pero quizás sería mejor distribuirla por masa. Sin embargo, en el código actual se aplica la misma fuerza a cada voxel.
    for body in voxel_bodies:
        p.applyExternalForce(body, -1, 
                           forceObj=fuerza_arrastre_total.tolist(), 
                           posObj=[0, 0, 0], 
                           flags=p.LINK_FRAME)

# Calcular la longitud acumulada de una trayectoria 3D
def longitud_acumulada(trayectoria):
    """
    Calcula la longitud acumulada de una trayectoria 3D.
    
    Parámetros:
        trayectoria (np.ndarray): matriz de tamaño (N,3) con coordenadas [x, y, z].
    
    Retorna:
        np.ndarray: vector de tamaño N con la distancia acumulada desde el inicio hasta cada punto.
    """
    # Diferencias entre puntos consecutivos
    deltas = np.diff(trayectoria, axis=0)  # tamaño (N-1,3)
    
    # Normas de cada vector delta (distancia entre puntos consecutivos)
    distancias = np.linalg.norm(deltas, axis=1)
    
    # Longitud acumulada, insertando 0 al inicio
    longitud = np.insert(np.cumsum(distancias), 0, 0)
    
    return longitud

def obtener_densidad_region(pos):
    """Obtiene la densidad de contaminantes en la región que contiene la posición"""
    region = tuple((np.array(pos) // TAMANO_REGION).astype(int))
    return densidad_contaminantes_regiones.get(region, DENSIDAD_INICIAL_CONTAMINANTES)

def actualizar_densidad_region(pos, particulas_capturadas):
    """Reduce la densidad en una región después de capturar partículas"""
    region = tuple((np.array(pos) // TAMANO_REGION).astype(int))
    if region not in densidad_contaminantes_regiones:
        densidad_contaminantes_regiones[region] = DENSIDAD_INICIAL_CONTAMINANTES
    
    # Reducir densidad proporcionalmente a las partículas capturadas
    volumen_region = TAMANO_REGION**3
    densidad_contaminantes_regiones[region] = max(0, densidad_contaminantes_regiones[region] - 
                                                 particulas_capturadas / volumen_region)

def aplicar_fuerza_arrastre_con_captura():
    """Aplica fuerza de arrastre y simula captura de partículas contaminantes"""
    global masa_micromaquina  # Declarar que usamos la variable global
    
    # Calcular velocidad y posición del centro de masa
    vel_total = np.zeros(3)
    pos_total = np.zeros(3)
    for body in voxel_bodies:
        vel, _ = p.getBaseVelocity(body)
        pos, _ = p.getBasePositionAndOrientation(body)
        vel_total += np.array(vel)
        pos_total += np.array(pos)
    
    vel_promedio = vel_total / len(voxel_bodies)
    pos_promedio = pos_total / len(voxel_bodies)
    
    # Obtener densidad en la posición actual
    densidad_actual = obtener_densidad_region(pos_promedio)
    
    # Calcular partículas potencialmente capturables
    tasa_captura_potencial = densidad_actual * AREA_EFECTIVA_CAPTURA * np.linalg.norm(vel_promedio)
    particulas_capturables = tasa_captura_potencial * (1/240)  # partículas por paso de simulación
    
    # Simular captura (probabilística)
    particulas_capturadas = np.random.poisson(particulas_capturables * EFICIENCIA_CAPTURA)
    
    if particulas_capturadas > 0:
        # Actualizar masa de la micromáquina
        masa_adicional = particulas_capturadas * MASA_POR_PARTICULA
        masa_micromaquina += masa_adicional
        
        # Actualizar densidad en la región
        actualizar_densidad_region(pos_promedio, particulas_capturadas)
        
        #print(f"Capturadas {particulas_capturadas} partículas. Masa actual: {masa_micromaquina:.2e} kg")
    
    # Fuerza de arrastre hidrodinámica (Stokes)
    radio_efectivo = (3 * masa_micromaquina / (4 * np.pi * 1000))**(1/3)  # Radio equivalente asumiendo densidad agua
    fuerza_arrastre_hidro = -6 * math.pi * VISCOSIDAD_AGUA * radio_efectivo * vel_promedio
    
    # Fuerza de arrastre adicional por contaminantes (depende de la densidad local)
    fuerza_arrastre_contaminantes = -COEFICIENTE_ARRASTRE_CONTAMINANTES * densidad_actual * vel_promedio
    
    # Fuerza total de arrastre
    fuerza_arrastre_total = fuerza_arrastre_hidro + fuerza_arrastre_contaminantes
    
    # Aplicar fuerza a todos los voxels
    for body in voxel_bodies:
        p.applyExternalForce(body, -1, 
                           forceObj=fuerza_arrastre_total.tolist(), 
                           posObj=[0, 0, 0], 
                           flags=p.LINK_FRAME)
# Bucle principal de simulación coon generacion y cada simulación
#primer bucle es para las micromáquinas
#segundo bucle es para cada una de las simulaciones
for m in range(5,6):
    #creamos las variabe
    vels_sistem_gral=[]
    tray_sistem_gral=[]
    densidad_contaminantes_regiones = {}
    for i in range(1,16):
        #m = m + 1
        #conectarse al simulador
        p.connect(p.GUI)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1/240)
        #configuración del entorno
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # Configura la gravedad
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        config_cubos = np.zeros((10, 10, 10))

        #para guardar el historial de todo el sistema
        voxels_hist = []
        voxel_bodies = []

        force_vector = np.array([1, -1, 0]) 
    
        #cargamos la micromáquina en el nuevo formato
        #'/Users/javimalikian/Documents/Tesis Master/Tesis Master/Algoritmos/PeerJ code/M1.json'
        ruta_micromaquina = PATH_GRAL + f"M{m}/M{m}.json"
        voxels_parametros, micromachine_id = load_micromachine_new_format(ruta_micromaquina)
        
        # Crear mapping de posición a body index
        pos_to_body = {tuple(voxel["p"]): idx for idx, voxel in enumerate(voxels_parametros)}

        # Creamos los voxels en el programa
        voxel = []
        for params in voxels_parametros:
            voxel_t = crete_voxel(posicion=params["p"], color=params["c"])
            voxel.append(voxel_t)
        
        #después de crear todos los voxels
        for body in voxel_bodies:
                p.changeDynamics(body, -1, 
                    linearDamping=0.8,    # Amortiguamiento lineal (0-1)
                    angularDamping=0.9,   # Amortiguamiento angular (0-1)
                    contactDamping=0.5,   # Amortiguamiento en contactos
                    contactStiffness=1e5) # Rigidez de contactos
            
        #creamos los constraints dependiendo de si están o no juntos
        crea_constraints()
        #inicializamos la masa de la micromáquina para cada simulación
        masa_micromaquina = 3e-13 * len(voxel)


        #iniciamos la simulación
        current_step = 0
        start = time.time()
        duracion = 60  # segundos
        vels_sistem = []
        tray_sistem = []
        
       
        #bucle de simulación de cada micromáquina
        while True:
            #avanzar la simulación
            p.stepSimulation()
            time.sleep(1./240.)  # espera

            # aplicamos fuerzas a las microalgas según sus vectores
            aplicar_fuerzas_alga()
             # aplicamos fuerza de arrastre (a TODOS los voxels)
            #aplicar_fuerza_arrastre_sin_contaminantes()
            #aplicar_fuerza_arrastre_con_contaminantes()
            aplicar_fuerza_arrastre_con_captura()
            
            #obtenemos la posición y velocidad del sistema
            pos, _ = p.getBasePositionAndOrientation(voxel[1])
            vel_l = get_micromachine_velocity(voxel)
            
            # Calculamos y mostramos velocidad
            vel = obt_vel_lineal(vel_l)
            print(f"m={m}, Simulation:{i} Paso {current_step}: Velocidad = {vel:.3f} {'μm/s'}", end="\r", flush=True)
            
            # Guardamos parámetros
            vels_sistem.append(vel)
            tray_sistem.append(pos)
            current_step += 1
            # Condición de parada por tiempo
            if time.time() - start >= duracion:
                break

        vels_sistem_gral.append(vels_sistem)
        tray_sistem_gral.append(tray_sistem)

        # Calculamos los parámetros y los enviamos a J()
        V = np.mean(vels_sistem)  # velocidad media
        M = 150e-11 * len(voxel)   # Masa dependiendo del numero de voxels
        #ecm, distancia_um = calc_error_cuad_medio(tray_sistem)
        
        print("---" * 20)
        print(f"Micromáquina ID: {micromachine_id}")
        print(f"Velocidad promedio: {V:.2f} μm/s")
        print(f"Masa total: {M:.2e} kg")
        #print(f"Error cuadrático medio: {ecm:.2e} m²")
        #print(f"Distancia al objetivo: {distancia_um:.2f} μm")
        print(f"Número de microalgas: {n_microalgas}")
        
        p.disconnect()
        #calculamos la frecuencia de muestreo real
        frecuencia_muestreo = current_step / duracion  
        # Paso de tiempo (dt):
        dt = 1 / frecuencia_muestreo

        #guardamos los datos con timestamps
        save_data_time(
            m,
            vels_sistem,
            tray_sistem,
            dt=dt,
            filename=f"M{m}/m_{m}_simulacion{i}_{micromachine_id}.json")
        
        # Crear gráficos
        archivo_simulacion = PATH_GRAL+ f"M{m}/m_{m}_simulacion{i}_{micromachine_id}.json"
        plot_trajectory_from_json(i,m,f'M{m}/m_{m}_simulacion{i}_{micromachine_id}.json')

        #borramos las variables para la siguiente simulación
        vels_sistem=[]
        tray_sistem=[]

    save_general_simulation_data(vels_sistem_gral, tray_sistem_gral, filename=f"M{m}/{m}_res_15_simulaciones_sin_carga.json")