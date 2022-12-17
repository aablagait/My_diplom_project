# импорт всех необходимых бибилиотек
import os
import numpy as np
import cv2
from yolo_utils import *
# import tensorflow as tf
from prediction import prediction


# основная функция, предназначенная для детектирования и предсказания
# дорожных знаков и светофоров
def detection(img):
    # создание имен классов
    classNames = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']

    # путь к архитектуре и весам модели
    modelConfig_path = 'config/worked.cfg'
    modelWeights_path = 'config/worked.weights'

    # загрузка архитектуры и весов модели с помощью CV2
    neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)

    # установка настроек графического процессора
    neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # определение размера входного кадра на вход нейронной сети
    network = neural_net
    height, width = 416, 416

    # доверительный и не максимальный порог подавления для этой версии YoloV3
    confidenceThreshold = 0.5
    nmsThreshold = 0.2

    # преобразование входного кадра в формат blob для OpenCV DNN
    outputs = convert_to_blob(img, network, height, width)

    # функция для детектирования объектов
    bounding_boxes, class_objects, confidence_probs = object_detection(outputs, img, confidenceThreshold)

    # функция, отбрасывающая объекты ниже парога non-max
    indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)

    # функция, предсказывающая цвет светофора или ограничение скорости
    classification_list = prediction(img, indices, bounding_boxes, class_objects)

    # отрисовка ограничивающих прямоугольников
    box_drawing(img, indices, bounding_boxes, class_objects, confidence_probs, classNames, classification_list, color=(255,0,0), thickness=2)

    # сохранение результата работы программы в отдельный файл
    img_save = cv2.imwrite('result.jpg', img)

    while True:
        # отображение результатов работы в отдельном окне
        cv2.imshow('Object detection in images', img)
        if cv2.waitKey(1) == 27:
            break

    # принудительное закрытие всех окон после окончание работы программа
    cv2.destroyAllWindows()

