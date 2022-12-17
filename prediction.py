# импорт необходимых библиотек и модулей
import tensorflow as tf
import numpy as np
import cv2


# функция prediction предназначенная для определения цвета светофора и значений знаков
# ограничения скорости
def prediction(img, indices, bounding_boxes, class_objects,):
    classification_list = []

    for i in indices:

        final_box = bounding_boxes[i]
        x, y, w, h = final_box[0], final_box[1], final_box[2], final_box[3]
        x, y, w, h = int(x), int(y), int(w), int(h)
        image = img[y:y + h, x:x + w]

        try:
            if class_objects[i] == 3:

                class_names = ['green', 'red', 'turn_off', 'wolked_green', 'wolked_red', 'yellow']
                model = tf.keras.models.load_model('config/traffic_light_6_classes.h5')
                image = tf.image.resize(image, [100, 100])
                # img = tf.keras.utils.load_img('images/traffic_light.png', target_size=(100, 100))
                img_array = tf.keras.utils.img_to_array(image)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model(img_array)
                score = tf.nn.softmax(predictions[0])

                result = class_names[np.argmax(score)], int(100 * np.max(score))
                classification_list.append(result)
                print(result)

            elif class_objects[i] == 1:

                class_names = ['40', '50', '60', '70', '80']
                model = tf.keras.models.load_model('config/limited_signs.h5')
                image = tf.image.resize(image, [100, 100])
                # img = tf.keras.utils.load_img('images/traffic_light.png', target_size=(100, 100))
                img_array = tf.keras.utils.img_to_array(image)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model(img_array)
                score = tf.nn.softmax(predictions[0])

                result = class_names[np.argmax(score)], int(100 * np.max(score))
                classification_list.append(result)
                print(result)

            else:
                print('знак пешеходного или стоп')
                classification_list.append((' ', ' '))

        except:
            print('error')

    return classification_list

