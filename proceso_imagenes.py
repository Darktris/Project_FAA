import os
import matplotlib.image as mpimg
import numpy as np
from copy import deepcopy
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.filters import median as filt_med
from skimage.morphology import disk
from scipy.signal import resample as rs

__umbral__ = 122
__disk_size__= 21
__ancho_ventana__=10
__n_bins__ = 64

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

def intensidad_mediana(image):
    return np.median(image)

def umbralizar_imagen(imagen):
    imagen_2 = deepcopy(imagen)

    imagen_2[imagen>__umbral__] = 255
    imagen_2[imagen<=__umbral__] = 0

    return filt_med(imagen_2, disk(__disk_size__))

def intensidad_media_umbralizada(imagen):
    return intensidad_media(umbralizar_imagen(imagen))

def intensidad_mediana_umbralizada(imagen):
    return intensidad_mediana(umbralizar_imagen(imagen))

def obtener_esqueleto(imagen):

    image_2 = skeletonize_3d((255 - umbralizar_imagen(imagen)) / 255)
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


def proyeccion_y(imagen):
    umbralizada = umbralizar_imagen(imagen)

    return np.fromiter(map(np.sum,255 - umbralizada),dtype=int)


def proyeccion_x(imagen):
    return proyeccion_y(imagen.T)

def alto_numero(imagen):
    proyeccion = proyeccion_y(imagen)

    index_floor = 0
    index_top = imagen.shape[0]

    for index ,p in enumerate(proyeccion):
        if p != 0:
            index_floor = index
            break

    for index ,p in enumerate(proyeccion[::-1]):
        if p != 0:
            index_top = imagen.shape[0] - index
            break

    return index_top -index_floor

def ancho_numero(imagen):
    return alto_numero(imagen.T)

def convolucion(histograma):
    return np.convolve(histograma, np.ones(shape=(__ancho_ventana__,)) / __ancho_ventana__,mode='same')


def estirar(histograma):
    index_left = 0
    index_right = 0

    for index, valor in enumerate(histograma):
        if valor > 0:
            index_left = index
            break

    for index, valor in enumerate(reversed(histograma)):
        if valor > 0:
            index_right = histograma.shape[0]-index
            break

    return histograma[index_left:index_right]

def resample(histograma,n_bins = __n_bins__):
        return rs(histograma,n_bins),n_bins

def estandarizar(histograma):
    return resample(estirar(histograma))[0]















