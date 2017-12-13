import os
import matplotlib.image as mpimg
import numpy as np
from copy import deepcopy
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.filters import median as filt_med
from skimage.morphology import disk

__umbral__ = 122

def cargar_numeros(folder):
    files = os.listdir(folder)
    images = []
    clases = []

    for file in files:
        images.append(mpimg.imread("./nums/" + file,True))
        clase = int(file[-5])
        clases.append(clase)

    return images, clases

def intensidad_media(image):
    return np.mean(image)

def umbralizar_imagen(imagen):
    imagen_2 = deepcopy(imagen)

    imagen_2[imagen>__umbral__] = 255
    imagen_2[imagen<=__umbral__] = 0

    return imagen_2

def intensidad_media_umbralizada(imagen):
    return intensidad_media(umbralizar_imagen(imagen))

def obtener_esqueleto(imagen):
    gray = filt_med(imagen, disk(7))

    image_2 = skeletonize_3d((255 - umbralizar_imagen(gray)) / 255)
    image_2[:30, :] = 0
    image_2[-30:, :] = 0
    image_2[:, :30] = 0
    image_2[:, -30:] = 0

    return image_2

def contar_rectas(imagen):
    esqueleto = obtener_esqueleto(imagen)
    h, theta, d = hough_line(esqueleto)
    _array, _, _ = hough_line_peaks(h, theta, d)

    return _array.shape[0]





