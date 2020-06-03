import cv2
import sys
import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal as dc
from random import randint

_LINE_COLOR = (24, 12, 248)
_CIRCLE_COLOR = (242, 0, 0)
_RECTANGLE_COLOR = (75, 13, 135)
_SOLID_BACK_COLOR = (81, 81, 81)
_EB_HEIGHT = 1080
_EB_WIDTH = 1080
_LENGTH_CRITERION = 91 #representa 1.5 metros en pixeles
_HEIGHt_TOP_PADDING = 320 #altura del relleno superior 320

#Calcula la distancia entre dos puntos
#dentro de la lista
def distance_points(first, second, list):

    distance = math.sqrt(
                    (list[first,0] - list[second,0])**2 +
                    (list[first,1] - list[second,1])**2)

    return distance

#Funcion recursiva para dibujar lineas entre todos los
#elementos de una lista de puntos (que representan a las
#personas) y al mismo tiempo valida el distanciamiento.
def draw_line_by_criterion(start_position, points, image):

    end_position = points.shape[0] - 1

    if start_position != end_position:
        while end_position > start_position:
            distance = distance_points(start_position, end_position, points)
            #solo dibuja la linea si la distancia es menor al criterio
            if distance <= _LENGTH_CRITERION:
                cv2.line(
                    image,
                    tuple(points[start_position]),
                    tuple(points[end_position]),
                    _LINE_COLOR,
                    5)

            end_position = end_position - 1

        draw_line_by_criterion(start_position + 1, points, image)

    return image

#Obtiene la matriz de transformacion
#width, height: dimensiones de la nueva perspectiva
def matrix_perspective():

    #Puntos que conforman el cuadrilatero origen y destino
    #los puntos de origen fueron calculados de forma experimental, dentro de varias
    #opciones se observo que este origen es la mejor opcion ademas de aprovechar
    #la mayor cantidad de puntos en las imagenes leidas.
    quadrilateral_s = np.float32([ [1152, 120],  [1914, 450], [1328, 1390], [144, 686]])
    #quadrilateral_s = np.float32([[976, 0], [1920, 169], [1364, 1068], [0, 442]])
    quadrilateral_t = np.float32([[0, 0], [_EB_WIDTH, 0], [_EB_WIDTH, _EB_HEIGHT], [0, _EB_HEIGHT]])

    #Obtener matriz de transformacion
    matrix_p = cv2.getPerspectiveTransform(src=quadrilateral_s , dst=quadrilateral_t)

    return matrix_p

#Crea la vista eye-bird, transforma la lista
#de puntos y para mostrarlos como circulos
def create_eye_bird(points, matrix_t):

    #Transformar puntos a la nueva perspectiva
    length = points.shape[0]
    points = points.reshape(1, length, 2)
    points_t = cv2.perspectiveTransform(np.float32(points), matrix_t)[0]

    #Crear imagen para la vista eye-bird
    img_eb = np.zeros((_EB_WIDTH, _EB_HEIGHT, 3))
    img_eb[:] = _SOLID_BACK_COLOR

    #Dibujar los puntos en la imagen
    for i in range(0, length):
        cv2.circle(
            img=img_eb,
            center=tuple(points_t[i].astype(int)),
            radius=12,
            color= _CIRCLE_COLOR,
            thickness=-1)

    return img_eb, points_t


#Leemos el video de entrada
video_input = cv2.VideoCapture('video/TownCentreXVID.avi')

video_input_second = 0
frame_rate = 1/25

#declaramos el video de salida
video_output = cv2.VideoWriter(
            'video/TownCentreXVID_output.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            1/frame_rate,
            (3000, 1080))

#obtenemos la matriz de transformacion
matrix_t = matrix_perspective()

#abrimos el arhivo
with open("input-csv/TC.csv", mode='r') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    #leemos el video en el segundo 0
    video_input.set(cv2.CAP_PROP_POS_MSEC, video_input_second)
    _ , img = video_input.read()
    count = 0
    #arreglo que contiene los centros de los bounding box
    points  = []

    #para cada una de las filas del archivo csv
    for row in csv_reader:

        #Valida cuando se llega al cambio de frames
        if int(row["numFrame"]) != count:

            #creamos la vista eye bird
            img_eb, points_t = create_eye_bird(np.array(points), matrix_t)

            #realizamos la validacion del distanciamiento y
            #dibujamos con lineas los puntos que no cumplen
            #con el criterio establecido
            img_eb_line = draw_line_by_criterion(0, np.array(points_t), img_eb)

            #unimos la imagen original con la vista de eye bird
            img_join = np.concatenate((img, img_eb_line), axis=1)
            video_output.write(np.uint8(img_join))

            #leemos la imagen correspondiente al siguiente frame
            video_input_second = round(video_input_second + frame_rate, 2)
            video_input.set(cv2.CAP_PROP_POS_MSEC, video_input_second*1000)
            _ , img = video_input.read()

            count = count + 1
            points.clear()

        #dibujamos los bounding box de la fila correspondiente al csv
        cv2.rectangle(
            img,
            (dc(row["bl"]), dc(row["bt"])),
            (dc(row["br"]), dc(row["bb"])),
            _RECTANGLE_COLOR,
            2)

        #calculamos el centro del rectangulo para representar
        #a una persona como punto
        x = (dc(row["bl"]) + dc(row["br"]))/2
        y = (dc(row["bt"]) + dc(row["bb"]))/2

        #para este caso se aplico un relleno superior para ampliar
        #la perspectiva por lo tanto se debe de actualizar la
        #altura de los puntos
        y = y + _HEIGHt_TOP_PADDING

        points.append([x, y])

    video_output.release()
