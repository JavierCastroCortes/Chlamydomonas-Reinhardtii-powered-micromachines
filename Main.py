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



PATH_GRAL = ""
VISCOSIDAD_AGUA = 8.9e-4  # Pa·s (viscosidad a 25°C)
VOXEL_SIZE = 10e-6
RADIO_CARACTERISTICO_INICIAL = VOXEL_SIZE / (2**(1/3)) # 30 μm - radio INICIAL (aumentado AGRESIVAMENTE para más arrastre)
DENSIDAD_VOXEL = 1200  # kg/m³ - densidad estimada de material de voxels

# ESTRATEGIA: Con radio = 30 μm, el arrastre Stokes será:
# F = -6πηrv = -6π × 8.9e-4 × 30e-6 × v = -5.02e-7 × v
# Para F = 100e-12 N (propulsión M2):
# 100e-12 = 5.02e-7 × v → v = 199 μm/s (más realista)

voxels_hist_general = []
fitness_hist = []

mejor_fitness = float('inf')
generaciones_sin_mejora = 0
m = 0
generacion = 0
n_microalgas = 0
masa_micromaquina = 0  # Se inicializa en el loop: 3e-13 * len(voxel_bodies)

# Parámetros de los voxels
masa = 15e-10
soporte_c = [0.5, 0.5, 0.5, 1]
alga_c = [0, 1, 0, 1]
cubo_size = 0.5

# Parámetros de fuerza de la microalga
alga_force = 10e-12  # Fuerza en Newtons (consistente con literatura C. reinhardtii)
alga_force_ex = 8e-12  # Componente extra para aleatoriedad

# PARÁMETROS DE CONTAMINANTES Y CAPTURA
# ========================================
DENSIDAD_INICIAL_CONTAMINANTES = 1e12  # partículas/m³
EFICIENCIA_CAPTURA = 0.1  # Probabilidad de captura (0-1)
AREA_EFECTIVA_CAPTURA = (10e-7)**2  # Área efectiva de captura (m²) = 1e-12 m²
MASA_POR_PARTICULA = 2.4e-15  # kg/partícula
TAMANO_REGION = 0.1  # metros - tamaño de regiones espaciales para tracking de densidad

# Coeficiente de arrastre por suspensión (bajo Reynolds)
# Basado en: F_arrastre_suspension = -C × ρ × A × v
# C ~ 0.1-0.5 para bajo Reynolds (dependiente de geometría)
COEFICIENTE_ARRASTRE_SUSPENSION = 0.1  # Coeficiente adimensional

# Diccionario para rastrear densidad por región (simula un campo espacial)
densidad_contaminantes_regiones = {}

# ELIMINADA: Esta línea causaba inconsistencia
# masa_micromaquina era 120e-11 aquí pero se sobrescribía como 3e-13*len(voxel) en línea 437
# Usamos SOLO la inicialización en el loop para consistencia


def load_micromachine_new_format(filename):
    """
    Carga el cromosoma en el nuevo formato JSON
    
    FORMATO ESPERADO:
    {
      "id": "0",
      "voxels": [
        {"p": [0,0,0], "c": 0, "f": [1,-1,0]},
        {"p": [1,0,0], "c": 1},
        ...
      ]
    }
    
    Args:
        filename: Ruta al archivo JSON
        
    Returns:
        voxels_parametros: Lista de voxels con parámetros
        chromosome_id: ID del cromosoma
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Archivo no encontrado: {filename}")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    voxels_parametros = data["voxels"]
    chromosome_id = data.get("id", "unknown")
    
    print(f"Cargado micromaquina ID: {chromosome_id} con {len(voxels_parametros)} voxels")
    return voxels_parametros, chromosome_id


def crete_voxel(masa=120e-10, tamano=0.5, posicion=[0, 0, 0], color=1):
    """
    Crea un voxel en la simulación PyBullet
    
    Args:
        masa: Masa del voxel en kg
        tamano: Tamaño del voxel (dimensión del cubo)
        posicion: Posición [x, y, z] del voxel
        color: 0 = verde (microalga), 1 = gris (estructura)
    """
    global n_microalgas
    
    color_c = [0.5, 0.5, 0.5, 1] if color == 1 else [0, 1, 0, 1]
    
    if color == 0:
        n_microalgas = n_microalgas + 1
    
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tamano, tamano, tamano])
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tamano, tamano, tamano], rgbaColor=color_c)
    body = p.createMultiBody(
        baseMass=masa, 
        baseCollisionShapeIndex=shape, 
        baseVisualShapeIndex=visual, 
        basePosition=posicion
    )
    voxel_bodies.append(body)
    return body


def crea_constraints():
    """
    Crea constraints FIXED entre voxels adyacentes para mantener estructura
    """
    voxel_size = 1.0  # Separación entre centros de voxels
    for i, voxel_i in enumerate(voxels_parametros):
        pos_i = np.array(voxel_i["p"])
        for j, voxel_j in enumerate(voxels_parametros[i+1:], start=i+1):
            pos_j = np.array(voxel_j["p"])
            distance = np.linalg.norm(pos_i - pos_j)
            
            # Verifica si son adyacentes (distancia = 1)
            if distance == 1.0:
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
    """
    Aplica fuerzas de propulsión según los vectores especificados en el JSON.
    
    FORMULACIÓN:
    - Fuerza base: F_base = f_vector × alga_force
    - Perturbaciones estocásticas: En ejes donde f[i] == 0 (complementarios)
    - Magnitud de perturbación: alga_force_ex (15e-12 N)
    
    FÍSICA:
    - Simula variabilidad biológica de flagelos de C. reinhardtii
    - Magnitud (~25 pN) consistente con mediciones experimentales
    """
    for idx, voxel in enumerate(voxels_parametros):
        if voxel["c"] == 0 and "f" in voxel:
            # Fuerza base según vector direccional
            force_base = np.array(voxel["f"]) * alga_force
            
            # Perturbaciones estocásticas solo en ejes complementarios
            random_vec = np.zeros(3)
            for i in range(3):
                if voxel["f"][i] == 0:
                    # Bernoulli con p=0.4 para +, 0.4 para -, 0.2 para 0
                    random_vec[i] = random.choices(
                        [alga_force_ex, -alga_force_ex, 0], 
                        weights=[0.4, 0.4, 0.2], 
                        k=1
                    )[0]
            
            force_vector = force_base + random_vec
            p.applyExternalForce(
                voxel_bodies[idx], -1, 
                forceObj=force_vector.tolist(), 
                posObj=[0, 0, 0], 
                flags=p.LINK_FRAME
            )


def obt_vel_lineal(vel_lin):
    """
    Obtiene la magnitud de velocidad y la convierte a μm/s
    
    Args:
        vel_lin: Vector velocidad en m/s
        
    Returns:
        speed: Magnitud de velocidad en μm/s
    """
    vx, vy, vz = vel_lin
    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    speed *= 1e6  # Convertir a μm/s
    return speed


def calc_error_cuad_medio(posiciones_simuladas):
    """
    Calcula error cuadrático medio y distancia al objetivo final
    """
    ruta_trayectoria = PATH_GRAL + "Pybullet/trayectoria_lineal.json"
    try:
        with open(ruta_trayectoria) as f:
            data = json.load(f)
        
        trayectoria_objetivo = [
            tuple(np.array(p["posicion_um"]) * 1e-6) 
            for p in data["trayectoria"]
        ]
        
        N = min(len(posiciones_simuladas), len(trayectoria_objetivo))
        sim = np.array(posiciones_simuladas[:N])
        ref = np.array(trayectoria_objetivo[:N])
        
        ecm = np.mean(np.sum((sim - ref)**2, axis=1))
        
        ultimo_objetivo_um = data["trayectoria"][-1]["posicion_um"]
        ultimo_objetivo_m = np.array(ultimo_objetivo_um) * 1e-6
        ultimo_simulado_m = np.array(posiciones_simuladas[-1])
        distancia_m = np.linalg.norm(ultimo_objetivo_m - ultimo_simulado_m)
        distancia_um = distancia_m * 1e6
        
        return ecm, distancia_um
        
    except FileNotFoundError:
        print(f"Archivo de trayectoria no encontrado: {ruta_trayectoria}")
        return float('inf'), float('inf')


def get_micromachine_velocity(voxel_bodies):
    """
    Obtiene la velocidad promedio de la micromáquina (centro de masa)
    
    UNIDADES: m/s
    
    Args:
        voxel_bodies: Lista de IDs de cuerpos en PyBullet
        
    Returns:
        avg_velocity: Vector velocidad promedio [vx, vy, vz] en m/s
    """
    velocities = []
    for voxel in voxel_bodies:
        vel, _ = p.getBaseVelocity(voxel)
        velocities.append(np.array(vel))
    
    avg_velocity = np.mean(velocities, axis=0)
    return avg_velocity


def save_data_time(m, vels_sistem, tray_sistem, dt=1/240, filename="simulacion_con_tiempo.json"):
    """
    Guarda datos de simulación con timestamps en formato JSON
    
    ESTRUCTURA DEL JSON:
    {
        "metadata": {
            "micromasquina": int,
            "frecuencia_muestreo_hz": float,
            "dt_segundos": float,
            "duracion_total_segundos": float
        },
        "datos": [
            {
                "tiempo_s": float,
                "velocidad_µm_s": float,
                "posicion_m": [float, float, float]
            }
        ]
    }
    
    Args:
        m: Número de micromáquina
        vels_sistem: Lista de velocidades (μm/s)
        tray_sistem: Lista de posiciones (m)
        dt: Paso de tiempo (s)
        filename: Ruta de archivo de salida
    """
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
    
    with open(filename, 'w') as f:
        json.dump(datos, f, indent=2)
    
    print(f"✓ Datos guardados en {filename}")


def obtener_densidad_region(pos):
    """
    Obtiene la densidad de contaminantes en la región que contiene la posición
    
    UNIDADES:
    - pos: posición en metros
    - Retorna: densidad en partículas/m³
    """
    region = tuple((np.array(pos) // TAMANO_REGION).astype(int))
    return densidad_contaminantes_regiones.get(region, DENSIDAD_INICIAL_CONTAMINANTES)


def actualizar_densidad_region(pos, particulas_capturadas):
    """
    Reduce la densidad en una región después de capturar partículas
    
    FÍSICA:
    - Reduce densidad proporcionalmente a partículas capturadas
    - Simula depleción local de contaminantes
    """
    region = tuple((np.array(pos) // TAMANO_REGION).astype(int))
    
    if region not in densidad_contaminantes_regiones:
        densidad_contaminantes_regiones[region] = DENSIDAD_INICIAL_CONTAMINANTES
    
    volumen_region = TAMANO_REGION**3
    reduccion_densidad = particulas_capturadas / volumen_region
    
    densidad_contaminantes_regiones[region] = max(
        0, 
        densidad_contaminantes_regiones[region] - reduccion_densidad
    )


def aplicar_fuerza_arrastre_sin_contaminantes():
    """
    ═══════════════════════════════════════════════════════════════════════════════
    FUNCIÓN: Aplica SOLO arrastre hidrostático (SIN partículas contaminantes)
    ═══════════════════════════════════════════════════════════════════════════════
    
    FÍSICA MODELADA:
    ────────────────
    Ley de Stokes para esfera en fluido viscoso:
    F_hidro = -6πηrv
    
    Donde:
    - η = viscosidad dinámica del agua (Pa·s)
    - r = radio efectivo de la micromáquina (m)
    - v = velocidad del centro de masa (m/s)
    
    UNIDADES:
    ─────────
    - Masa: kg
    - Velocidad: m/s
    - Fuerza: N
    
    DISTRIBUCIÓN:
    ─────────────
    F_por_voxel = F_total / N_voxels (CRÍTICO: dividir, NO multiplicar)
    """
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 1: CALCULAR VELOCIDAD Y POSICIÓN DEL CENTRO DE MASA
    # ════════════════════════════════════════════════════════════════════════════
    vel_total = np.zeros(3)  # m/s
    pos_total = np.zeros(3)  # m
    
    for body in voxel_bodies:
        vel, _ = p.getBaseVelocity(body)  # m/s
        pos, _ = p.getBasePositionAndOrientation(body)  # m
        vel_total += np.array(vel)
        pos_total += np.array(pos)
    
    vel_promedio = vel_total / len(voxel_bodies)  # m/s
    pos_promedio = pos_total / len(voxel_bodies)  # m
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 2: CALCULAR RADIO EFECTIVO (DINÁMICO)
    # ════════════════════════════════════════════════════════════════════════════
    
    # Radio basado en masa actual (esfera equivalente)
    radio_efectivo = (3 * masa_micromaquina / (4 * np.pi * DENSIDAD_VOXEL))**(1/3)  # m
    radio_efectivo = max(radio_efectivo, RADIO_CARACTERISTICO_INICIAL * 0.5)
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 3: CALCULAR FUERZA DE ARRASTRE HIDROSTÁTICO (LEY DE STOKES)
    # ════════════════════════════════════════════════════════════════════════════
    
    # F_hidro = -6πηrv
    fuerza_arrastre_total = (
        -6 * np.pi * VISCOSIDAD_AGUA * radio_efectivo * vel_promedio
    )  # N
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 4: DISTRIBUIR FUERZA ENTRE VOXELS
    # ════════════════════════════════════════════════════════════════════════════
    
    fuerza_por_voxel = fuerza_arrastre_total / len(voxel_bodies)
    
    for body in voxel_bodies:
        p.applyExternalForce(
            body, -1,
            forceObj=fuerza_por_voxel.tolist(),  # Vector fuerza [N]
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )


def aplicar_fuerza_arrastre_con_captura():
    """
    ═══════════════════════════════════════════════════════════════════════════════
    FUNCIÓN PRINCIPAL: Aplica fuerzas de arrastre y simula captura de partículas
    ═══════════════════════════════════════════════════════════════════════════════
    
    FÍSICA MODELADA:
    ────────────────
    1. ARRASTRE HIDROSTÁTICO (Ley de Stokes para esfera):
       F_hidro = -6πηrv
       
       Donde:
       - η = viscosidad dinámica (Pa·s)
       - r = radio efectivo de la micromáquina (m) [DINÁMICO]
       - v = velocidad del centro de masa (m/s)
    
    2. ARRASTRE POR SUSPENSIÓN (partículas contaminantes):
       F_suspension = -C × ρ × A × v
       
       Donde:
       - C = coeficiente adimensional (~0.1, bajo Reynolds)
       - ρ = densidad de partículas (partículas/m³) [LOCAL]
       - A = área proyectada (m²)
       - v = velocidad (m/s)
    
    3. CAPTURA DE PARTÍCULAS:
       - Tasa de captura depende de densidad local, velocidad y área
       - Eficiencia probabilística (parámetro)
       - Aumenta masa de la micromáquina → aumenta radio → aumenta arrastre
    
    UNIDADES CONSISTENTES:
    ─────────────────────
    - Masa: kg
    - Velocidad: m/s
    - Fuerza: N
    - Radio: m
    - Densidad: partículas/m³
    - Posición: m
    
    DISTRIBUCIÓN DE FUERZAS:
    ───────────────────────
    La fuerza total se divide EQUITATIVAMENTE entre todos los voxels:
    F_por_voxel = F_total / N_voxels
    
    ESTO ES CRÍTICO: Si se aplica F_total a cada voxel, la fuerza neta es N×F_total
    """
    global masa_micromaquina
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 1: CALCULAR VELOCIDAD Y POSICIÓN DEL CENTRO DE MASA
    # ════════════════════════════════════════════════════════════════════════════
    vel_total = np.zeros(3)  # m/s
    pos_total = np.zeros(3)  # m
    
    for body in voxel_bodies:
        vel, _ = p.getBaseVelocity(body)  # m/s
        pos, _ = p.getBasePositionAndOrientation(body)  # m
        vel_total += np.array(vel)
        pos_total += np.array(pos)
    
    # CORRECCIÓN: Cálculo fuera del loop (NO dentro)
    vel_promedio = vel_total / len(voxel_bodies)  # m/s
    pos_promedio = pos_total / len(voxel_bodies)  # m
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 2: OBTENER DENSIDAD LOCAL DE CONTAMINANTES
    # ════════════════════════════════════════════════════════════════════════════
    densidad_actual = obtener_densidad_region(pos_promedio)  # partículas/m³
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 3: SIMULAR CAPTURA DE PARTÍCULAS
    # ════════════════════════════════════════════════════════════════════════════
    
    # Tasa de captura potencial basada en:
    # - Densidad local (partículas/m³)
    # - Área efectiva de captura (m²)
    # - Magnitud de velocidad relativa (m/s)
    vel_magnitud = np.linalg.norm(vel_promedio)  # m/s
    tasa_captura_potencial = densidad_actual * AREA_EFECTIVA_CAPTURA * vel_magnitud
    
    # Ajustar por paso de simulación (240 Hz)
    # Convertir tasa (partículas/s) a partículas por timestep
    particulas_capturables = tasa_captura_potencial * (1.0 / 240.0)
    
    # Simular captura como proceso Poisson
    particulas_capturadas = np.random.poisson(particulas_capturables * EFICIENCIA_CAPTURA)
    
    if particulas_capturadas > 0:
        # Actualizar masa de la micromáquina
        masa_adicional = particulas_capturadas * MASA_POR_PARTICULA
        masa_micromaquina += masa_adicional
        
        # Actualizar densidad local (depleción)
        actualizar_densidad_region(pos_promedio, particulas_capturadas)
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 4: CALCULAR RADIO EFECTIVO (DINÁMICO)
    # ════════════════════════════════════════════════════════════════════════════
    
    # CORRECCIÓN #1: Radio DINÁMICO que cambia con masa capturada
    # Suponemos una esfera equivalente con la masa actual
    # V = (3/4)πr³ → r = (3V/4π)^(1/3) = (3M/4πρ)^(1/3)
    
    radio_efectivo = (3 * masa_micromaquina / (4 * np.pi * DENSIDAD_VOXEL))**(1/3)  # m
    
    # Validación: radio no debe ser menor que el radio inicial
    radio_efectivo = max(radio_efectivo, RADIO_CARACTERISTICO_INICIAL * 0.5)
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 5: CALCULAR FUERZA DE ARRASTRE HIDROSTÁTICO (LEY DE STOKES)
    # ════════════════════════════════════════════════════════════════════════════
    
    # F_hidro = -6πηrv
    # Unidades: [Pa·s] × [m] × [m/s] = [kg/(m·s)] × [m] × [m/s] = [kg·m/s²] = [N] ✓
    
    fuerza_arrastre_hidro = (
        -6 * np.pi * VISCOSIDAD_AGUA * radio_efectivo * vel_promedio
    )  # N
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 6: CALCULAR FUERZA DE ARRASTRE POR SUSPENSIÓN
    # ════════════════════════════════════════════════════════════════════════════
    
    # Modelar efecto de partículas suspendidas como arrastre adicional
    # F_suspension = -C × ρ × A × v
    # donde:
    # - C = coeficiente adimensional (bajo Reynolds: 0.1-0.5)
    # - ρ = densidad de partículas EN UNIDADES CORRECTAS
    # - A = área proyectada (m²)
    # - v = velocidad (m/s)
    
    # Conversión: densidad en partículas/m³ → densidad EFECTIVA en kg/m³
    # Asumiendo partículas esféricas de radio 1 μm:
    radio_particula = 1e-6  # m
    volumen_particula = (4/3) * np.pi * radio_particula**3  # m³
    densidad_masa_particulas = densidad_actual * MASA_POR_PARTICULA  # kg/m³
    
    # Área proyectada de la micromáquina (esfera)
    area_proyectada = np.pi * radio_efectivo**2  # m²
    
    # Fuerza de arrastre por suspensión
    fuerza_arrastre_suspension = (
        -COEFICIENTE_ARRASTRE_SUSPENSION * densidad_masa_particulas * area_proyectada * vel_promedio
    )  # N
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 7: FUERZA TOTAL DE ARRASTRE
    # ════════════════════════════════════════════════════════════════════════════
    
    fuerza_arrastre_total = fuerza_arrastre_hidro + fuerza_arrastre_suspension  # N
    
    # ════════════════════════════════════════════════════════════════════════════
    # PASO 8: DISTRIBUIR FUERZA ENTRE VOXELS (CORRECCIÓN #2)
    # ════════════════════════════════════════════════════════════════════════════
    
    # CRÍTICO: Dividir la fuerza entre número de voxels, NO multiplicar
    # Si hay 5 voxels y no se divide:
    #   Fuerza aplicada a cada voxel = F_total
    #   Fuerza neta = 5 × F_total (INCORRECTO, 5× demasiado fuerte)
    #
    # Dividiendo:
    #   Fuerza por voxel = F_total / 5
    #   Fuerza neta = 5 × (F_total/5) = F_total (CORRECTO)
    
    fuerza_por_voxel = fuerza_arrastre_total / len(voxel_bodies)
    
    for body in voxel_bodies:
        p.applyExternalForce(
            body, -1,
            forceObj=fuerza_por_voxel.tolist(),  # Vector fuerza [N]
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL DE SIMULACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

for m in range(1, 6):  # M1 a M5
    print(f"\n{'='*80}")
    print(f"INICIANDO SIMULACIONES PARA M{m}")
    print(f"{'='*80}\n")
    
    vels_sistem_gral = []
    tray_sistem_gral = []
    densidad_contaminantes_regiones = {}  # Reinicializar para cada M
    
    for i in range(1, 16):  # 15 simulaciones por micromáquina
        print(f"\n[{i}/15] Iniciando simulación...")
        
        # ────────────────────────────────────────────────────────────────────────
        # CONECTAR A PyBullet
        # ────────────────────────────────────────────────────────────────────────
        p.connect(p.GUI)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1/240)  # 240 Hz
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # ────────────────────────────────────────────────────────────────────────
        # INICIALIZAR VARIABLES
        # ────────────────────────────────────────────────────────────────────────
        config_cubos = np.zeros((10, 10, 10))
        voxels_hist = []
        voxel_bodies = []
        
        force_vector = np.array([1, -1, 0])
        
        # ────────────────────────────────────────────────────────────────────────
        # CARGAR MICROMÁQUINA
        # ────────────────────────────────────────────────────────────────────────
        ruta_micromaquina = PATH_GRAL + f"M{m}/M{m}.json"
        voxels_parametros, micromachine_id = load_micromachine_new_format(ruta_micromaquina)
        
        pos_to_body = {tuple(voxel["p"]): idx for idx, voxel in enumerate(voxels_parametros)}
        
        # ────────────────────────────────────────────────────────────────────────
        # CREAR VOXELS
        # ────────────────────────────────────────────────────────────────────────
        voxel = []
        for params in voxels_parametros:
            voxel_t = crete_voxel(posicion=params["p"], color=params["c"])
            voxel.append(voxel_t)
        
        # ────────────────────────────────────────────────────────────────────────
        # CONFIGURAR DINÁMICAS
        # ────────────────────────────────────────────────────────────────────────
        for body in voxel_bodies:
            p.changeDynamics(
                body, -1,
                linearDamping=0.8,
                angularDamping=0.9,
                contactDamping=0.5,
                contactStiffness=1e5
            )
        
        # ────────────────────────────────────────────────────────────────────────
        # CREAR CONSTRAINTS
        # ────────────────────────────────────────────────────────────────────────
        crea_constraints()
        
        # ────────────────────────────────────────────────────────────────────────
        # INICIALIZAR MASA (CONSISTENTE)
        # ────────────────────────────────────────────────────────────────────────
        masa_micromaquina = 1.5e-12 * len(voxel)  # kg - AUMENTADO 5× (de 3e-13 a 1.5e-12) para más arrastre
        
        # ────────────────────────────────────────────────────────────────────────
        # BUCLE DE SIMULACIÓN
        # ────────────────────────────────────────────────────────────────────────
        current_step = 0
        start = time.time()
        duracion = 60  # segundos
        vels_sistem = []
        tray_sistem = []
        
        while True:
            # Avanzar simulación
            p.stepSimulation()
            time.sleep(1./240.)
            
            # Aplicar fuerzas de microalga
            aplicar_fuerzas_alga()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # OPCIÓN 1: SIN CONTAMINANTES (línea base)
            # ═══════════════════════════════════════════════════════════════════════════
            #aplicar_fuerza_arrastre_sin_contaminantes()
            
            # ═══════════════════════════════════════════════════════════════════════════
            # OPCIÓN 2: CON CONTAMINANTES (descomenta esta línea para simular con carga)
            # ═══════════════════════════════════════════════════════════════════════════
            aplicar_fuerza_arrastre_con_captura()
            
            # Obtener posición y velocidad
            pos, _ = p.getBasePositionAndOrientation(voxel[0])
            vel_l = get_micromachine_velocity(voxel)
            
            # Calcular velocidad en μm/s
            vel = obt_vel_lineal(vel_l)
            
            # Mostrar progreso
            print(
                f"m={m}, Sim:{i:02d}, Paso {current_step:04d}: Vel={vel:7.3f} μm/s, "
                f"Masa={masa_micromaquina:.3e} kg",
                end="\r", 
                flush=True
            )
            
            # Guardar datos
            vels_sistem.append(vel)
            tray_sistem.append(pos)
            current_step += 1
            
            # Condición de parada
            if time.time() - start >= duracion:
                break
        
        vels_sistem_gral.append(vels_sistem)
        tray_sistem_gral.append(tray_sistem)
        
        # ────────────────────────────────────────────────────────────────────────
        # ESTADÍSTICAS
        # ────────────────────────────────────────────────────────────────────────
        V = np.mean(vels_sistem)
        M = 150e-11 * len(voxel)
        
        print("\n" + "─"*80)
        print(f"Micromáquina ID: {micromachine_id}")
        print(f"Velocidad promedio: {V:.2f} μm/s")
        print(f"Masa final: {M:.2e} kg")
        print(f"Número de microalgas: {n_microalgas}")
        print(f"Pasos totales: {current_step}")
        print("─"*80)
        
        # ────────────────────────────────────────────────────────────────────────
        # GUARDAR DATOS
        # ────────────────────────────────────────────────────────────────────────
        p.disconnect()
        
        frecuencia_muestreo = current_step / duracion
        dt = 1 / frecuencia_muestreo
        
        save_data_time(
            m,
            vels_sistem,
            tray_sistem,
            dt=dt,
            filename=f"M{m}/m_{m}_simulacion{i}_{micromachine_id}.json"
        )
        
        # Crear gráficos
        archivo_simulacion = PATH_GRAL + f"M{m}/m_{m}_simulacion{i}_{micromachine_id}.json"
        plot_trajectory_from_json(i, m, f'M{m}/m_{m}_simulacion{i}_{micromachine_id}.json')
        
        # Limpiar
        vels_sistem = []
        tray_sistem = []
        n_microalgas = 0  # Reinicializar contador
    
    # ────────────────────────────────────────────────────────────────────────────
    # GUARDAR RESULTADOS AGREGADOS
    # ────────────────────────────────────────────────────────────────────────────
    save_general_simulation_data(
        vels_sistem_gral, 
        tray_sistem_gral, 
        filename=f"M{m}/{m}_res_15_simulaciones_CON_CARGA_CORREGIDO.json"
    )

print("\n" + "="*80)
print("✓ TODAS LAS SIMULACIONES COMPLETADAS")
print("="*80)
