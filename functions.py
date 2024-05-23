import cv2
import numpy as np
from deepforest import main
from google.colab.patches import cv2_imshow

def detect_shadows(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen a espacio de color LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extraer el canal L (luminosidad)
    l_channel = lab[:,:,0]

    # Calcular un umbral adaptativo para identificar las áreas más oscuras
    _, shadow_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Dilatar la máscara de sombras para una mejor visualización
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

    # Calcular el área de sombra
    shadow_pixels = cv2.countNonZero(shadow_mask)
    total_pixels = image.shape[0] * image.shape[1]
    shadow_percentage = 100-(shadow_pixels / total_pixels) * 100

    # Aplicar la máscara de sombras a la imagen original
    result = cv2.bitwise_and(image, image, mask=shadow_mask)

    # Mostrar la imagen resultante con áreas de sombra resaltadas

    return result, shadow_mask, shadow_percentage

def detect_asphalt(image_path):    
    image = cv2.imread(image_path)

    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de colores para detectar carreteras (podrías ajustarlos según tu imagen)
    lower_color = np.array([85, 85, 125])
    upper_color = np.array([215, 215, 215])

    # Aplicar la máscara para detectar las áreas de carretera
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplicar operaciones morfológicas para eliminar el ruido
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos detectados sobre la imagen original
    street_image = image.copy()
    cv2.drawContours(street_image, contours, -1, (0, 255, 0), 2)
    # Calcular el porcentaje de área de carretera
    total_pixels = mask.shape[0] * mask.shape[1]
    street_pixels = cv2.countNonZero(mask)
    street_percentage = (street_pixels / total_pixels) * 100

    # Mostrar el porcentaje
    print("Porcentaje de área de carretera: {:.2f}%".format(street_percentage))
  return street_image, street_percentage


def detect_vegetation(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Instanciar y cargar el modelo DeepForest preentrenado
    model = main.deepforest()
    model.use_release()

    # Convertir la imagen de OpenCV (BGR) a formato RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Realizar predicciones de detección de árboles
    predictions = model.predict_image(np.array(image_rgb))

    # Contar el número de árboles
    num_trees = len(predictions)

    # Crear una máscara para la vegetación (en este caso, árboles)
    vegetation_mask = np.zeros_like(original_image, dtype=np.uint8)

    for index, row in predictions.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(vegetation_mask, (x1, y1), (x2, y2), (0, 255, 0), -1)

    # Calcular el porcentaje de área de vegetación
    total_pixels = original_image.shape[0] * original_image.shape[1]
    vegetation_pixels = cv2.countNonZero(cv2.cvtColor(vegetation_mask, cv2.COLOR_BGR2GRAY))
    vegetation_percentage = (vegetation_pixels / total_pixels) * 100

    # Superponer la máscara de vegetación sobre la imagen original
    result_image = cv2.addWeighted(original_image, 1, vegetation_mask, 0.5, 0)

    # Mostrar la imagen resultante con áreas de vegetación resaltadas
    cv2_imshow(result_image)

    print("Número de árboles detectados: {}".format(num_trees))
    print("Porcentaje de área de vegetación: {:.2f}%".format(vegetation_percentage))

    return result_image, num_trees,vegetation_percentage

