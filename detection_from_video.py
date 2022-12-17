import os
import numpy as np
import cv2
from yolo_utils import *
# import tensorflow as tf
from prediction import prediction


def detection_from_video(path):
    # load the obj/classes names
    classNames = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']

    # load the model config and weights
    modelConfig_path = 'config/worked.cfg'
    modelWeights_path = 'config/worked.weights'

    # read the model cfg and weights with the cv2 DNN module
    neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
    # set the preferable Backend to GPU
    neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # defining the input frame resolution for the neural network to process
    network = neural_net
    height, width = 416, 416

    # confidence and non-max suppression threshold for this YoloV3 version
    confidenceThreshold = 0.5
    nmsThreshold = 0.2

    video = cv2.VideoCapture(path)
    while True:
        _, img = video.read()
        # using convert_to_blob function :
        outputs = convert_to_blob(img, network, height, width)
        # apply object detection on the video file
        bounding_boxes, class_objects, confidence_probs = object_detection(outputs, img, confidenceThreshold)


        # perform non-max suppression
        indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)

        # test
        classification_list = prediction(img, indices, bounding_boxes, class_objects)

        # draw the boxes
        box_drawing(img, indices, bounding_boxes, class_objects, confidence_probs, classNames, classification_list, color=(255,0,0), thickness=2)

        # to save the detected image
        # img_save = cv2.imwrite('result.jpg', img)


        cv2.imshow('Object detection in images', img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
