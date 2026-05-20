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
