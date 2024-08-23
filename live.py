import cv2
import numpy as np
from dotenv import load_dotenv
import os

# Cargar las variables de entorno de ambos archivos .env
load_dotenv()  # Cargar el archivo .env
load_dotenv('cords.env')  # Cargar el archivo cords.env

# Obtener las coordenadas del área desde cords.env
area_width = int(os.getenv('AREA_WIDTH', 300))
area_height = int(os.getenv('AREA_HEIGHT', 300))
area_x_pos = int(os.getenv('AREA_X_POS', 0))
area_y_pos = int(os.getenv('AREA_Y_POS', 0))

# Obtener las teclas desde .env
key_expand_width = ord(os.getenv('KEY_EXPAND_WIDTH', 'e'))
key_shrink_width = ord(os.getenv('KEY_SHRINK_WIDTH', 'q'))
key_expand_height = ord(os.getenv('KEY_EXPAND_HEIGHT', '+'))
key_shrink_height = ord(os.getenv('KEY_SHRINK_HEIGHT', '-'))
key_exit = int(os.getenv('KEY_EXIT', 27))

# Abrir la webcam
cap = cv2.VideoCapture(0)  # 0 generalmente corresponde a la cámara predeterminada

# Crear el objeto de sustracción de fondo con parámetros ajustados
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=True)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Kernel para apertura
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel para cierre

def get_area_pts(x, y, width, height):
    return np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
    ])

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Mejorar el contraste

    # Calcular la posición inicial del área para que esté centrada en la parte inferior
    x_center = (frame.shape[1] - area_width) // 2
    y_bottom = frame.shape[0] - area_height - 10  # Dejar un margen de 10 píxeles desde la parte inferior
    x_pos = x_center + area_x_pos
    y_pos = y_bottom + area_y_pos

    # Obtener los puntos del área ajustados por el tamaño
    area_pts = get_area_pts(x_pos, y_pos, area_width, area_height)

    # Crear una imagen auxiliar y determinar el área a analizar
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    # Aplicar el sustractor de fondo y procesar la imagen
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)  # Aplicar apertura para reducir el ruido
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)  # Aplicar cierre para rellenar huecos
    fgmask = cv2.dilate(fgmask, None, iterations=2)  # Aumentar el número de iteraciones de dilatación

    # Encontrar contornos en fgmask
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    num_people = 0  # Contador de personas
    texto_estado = "Estado: No se ha detectado movimiento"
    color = (0, 255, 0)

    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:  # Ajustar el umbral para mayor precisión
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado = "Estado: Alerta Movimiento Detectado!"
            color = (0, 0, 255)
            num_people += 1

    # Visualizar el alrededor del área analizada y el estado de la detección
    cv2.drawContours(frame, [area_pts], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Personas detectadas: {num_people}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('fgmask', fgmask)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(70) & 0xFF
    if k == key_exit:  # Salir con la tecla definida en .env
        break
    elif k == key_expand_width:  # Expandir el área horizontalmente
        area_width += 10
    elif k == key_shrink_width:  # Reducir el área horizontalmente
        area_width = max(10, area_width - 10)
    elif k == key_expand_height:  # Expandir el área verticalmente
        area_height += 10
    elif k == key_shrink_height:  # Reducir el área verticalmente
        area_height = max(10, area_height - 10)

    # Actualizar las coordenadas en el archivo cords.env
    with open('cords.env', 'w') as f:
        f.write(f"AREA_WIDTH={area_width}\n")
        f.write(f"AREA_HEIGHT={area_height}\n")
        f.write(f"AREA_X_POS={area_x_pos}\n")
        f.write(f"AREA_Y_POS={area_y_pos}\n")

cap.release()
cv2.destroyAllWindows()
