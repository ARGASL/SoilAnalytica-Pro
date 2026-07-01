import cv2
import numpy as np

def procesar_cromatograma(path_imagen):
    img = cv2.imread(path_imagen)
    if img is None: return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Buscamos el círculo del papel filtro
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=50, param2=25, minRadius=150, maxRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0, 0] # Tomamos el círculo más probable
        mask = np.zeros_like(gray)
        cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
        # Recortamos con un margen pequeño para que se vea limpio
        res = cv2.bitwise_and(img, img, mask=mask)
        return img, res
    return img, None

# =========================================================================
# FUNCIONES ADICIONALES DE ANÁLISIS MORFOLÓGICO Y CROMÁTICO AVANZADO
# =========================================================================

def analizar_color_hsv_zonas(imagen, cy, cx, max_r):
    """
    Analiza los tonos reales (HSV) en la zona media para evitar pérdidas del gris.
    """
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    mask_media = (dist >= max_r * 0.35) & (dist <= max_r * 0.68)
    tonos_media = hsv[:, :, 0][mask_media] if np.any(mask_media) else []
    
    mineral_detectado = "Matriz balanceada / Cromatografía estándar"
    if len(tonos_media) > 0:
        hist, _ = np.histogram(tonos_media, bins=18, range=(0, 180))
        tono_dominante = np.argmax(hist) * 10
        
        if 10 <= tono_dominante <= 25:
            mineral_detectado = "Presencia de Óxidos de Hierro / Arcillas estables (Tonos pardos/rojizos)"
        elif 26 <= tono_dominante <= 45:
            mineral_detectado = "Alta concentración de Carbonatos / Silicatos activos (Tonos dorados/amarillentos)"
        elif tono_dominante < 10 or tono_dominante > 150:
            mineral_detectado = "Materia orgánica cruda / Acumulación de sulfuros (Tonos oscuros/púrpura)"

    return {"mineral_predominante": mineral_detectado}

def analizar_orillas_y_enzimas(imagen, cy, cx, max_r):
    """
    Detecta la rugosidad física del contorno usando Canny para inferir actividad enzimática.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    difuminado = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(difuminado, 30, 90)
    
    h, w = gris.shape
    mascara_orilla = np.zeros_like(gris)
    cv2.circle(mascara_orilla, (cx, cy), int(max_r * 1.05), 255, -1)
    cv2.circle(mascara_orilla, (cx, cy), int(max_r * 0.70), 0, -1)
    bordes_filtrados = cv2.bitwise_and(bordes, mascara_orilla)
    
    contornos, _ = cv2.findContours(bordes_filtrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return {"actividad_enzimatica": "Moderada (Borde regular continuo)"}
        
    c = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(c, True)
    area = cv2.contourArea(c) + 1e-6
    
    circularidad = (4 * np.pi * area) / (perimetro ** 2)
    rugosidad_score = np.clip((1.0 - circularidad) * 200, 10, 95)
    
    if rugosidad_score > 65:
        status = "Fuerte actividad enzimática (Borde muy festoneado/puntiagudo)"
    elif rugosidad_score > 35:
        status = "Actividad enzimática media (Borde ligeramente ondulado)"
    else:
        status = "Baja actividad enzimática / Bloqueo orgánico (Borde rígido y liso)"
        
    return {"actividad_enzimatica": status}

def analizar_plumas_cresta(imagen, cy, cx, max_r):
    """
    Desenrolla el círculo mediante una transformación polar para evaluar
    los patrones e irregularidades en forma de 'plumas'.
    """
    radio_destino = float(max_r)
    unwrapped = cv2.warpPolar(imagen, (360, int(radio_destino)), (cx, cy), radio_destino, cv2.WARP_POLAR_LINEAR)
    
    zona_media = unwrapped[int(max_r*0.35):int(max_r*0.75), :, :]
    gris_media = cv2.cvtColor(zona_media, cv2.COLOR_BGR2GRAY)
    perfil_angular = np.mean(gris_media, axis=0)
    
    picos = 0
    for i in range(1, len(perfil_angular) - 1):
        if perfil_angular[i] > perfil_angular[i-1] and perfil_angular[i] > perfil_angular[i+1]:
            if perfil_angular[i] - np.mean(perfil_angular) > 5:
                picos += 1
                
    if picos > 28:
        mineralizacion_tipo = f"Dinamismo mineral sobresaliente ({picos} zonas de flujo/plumas)"
    elif picos > 15:
        mineralizacion_tipo = f"Flujo mineral intermedio ({picos} plumas detectadas)"
    else:
        mineralizacion_tipo = f"Flujo mineral restringido o cementado ({picos} plumas)"
        
    return {"patron_plumas": mineralizacion_tipo}
