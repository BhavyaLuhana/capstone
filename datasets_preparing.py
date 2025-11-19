import pickle

# Importing other standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pylab import text
import csv
from PIL import Image
from skimage.transform import resize


def load_rgb_data(file):

    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        x = d['features'].astype(np.float32)  # images
        y = d['labels']   # labels
        s = d['sizes']    # sizes of image
        c = d['coords']    # coordinates of image
       
    return x, y, s, c

def rgb_to_gray_data(x_data):
    x_g = np.zeros((x_data.shape[0], 1, 32, 32))

    x_g[:, 0, :, :] = x_data[:, 0, :, :] * 0.299 + x_data[:, 1, :, :] * 0.587 + x_data[:, 2, :, :] 


    return x_g


def label_text(file):
    label_list = []

    with open(file, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            label_list.append(row[1])

        del label_list[0]

    return label_list


def brightness_changing(image):
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    random_brightness = 0.25 + np.random.uniform()
    
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * random_brightness
    
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image_rgb


def rotation_changing(image):
    angle_range = 25

    angle_rotation = np.random.uniform(angle_range) - angle_range / 2

    rows, columns, channels = image.shape

    affine_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), angle_rotation, 1)

    rotated_image = cv2.warpAffine(image, affine_matrix, (columns, rows))

    return rotated_image


def transformation_brightness_rotation(image):
    return brightness_changing(rotation_changing(image))


def random_image(x_train, y_train, y_number):
    
    image_indexes = np.where(y_train == y_number)
    
    random_index = np.random.randint(0, np.bincount(y_train)[y_number] - 1)
    
    return x_train[image_indexes][random_index]


def equalize_training_dataset(x_train, y_train):
    number_of_examples_for_every_label = np.bincount(y_train)
    number_of_labels = np.arange(len(number_of_examples_for_every_label))


    for i in tqdm(number_of_labels):
        number_of_examples_to_add = int(np.mean(number_of_examples_for_every_label) * 2.5) - \
                                    number_of_examples_for_every_label[i]

        x_temp = []
        y_temp = []

        for j in range(number_of_examples_to_add):
            getting_random_image = random_image(x_train, y_train, i)
            x_temp.append(transformation_brightness_rotation(getting_random_image))
            y_temp.append(i)

        x_train = np.append(x_train, np.array(x_temp), axis=0)
        y_train = np.append(y_train, np.array(y_temp), axis=0)

    return x_train, y_train


def local_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    return clahe.apply(image)


def preprocess_data(d, shuffle=False, lhe=False, norm_255=False, mean_norm=False, std_norm=False,
                    transpose=True, colour='rgb'):
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(d['x_train'])
        np.random.seed(0)
        np.random.shuffle(d['y_train'])
        np.random.seed(0)
        np.random.shuffle(d['x_validation'])
        np.random.seed(0)
        np.random.shuffle(d['y_validation'])
        np.random.seed(0)
        np.random.shuffle(d['x_test'])
        np.random.seed(0)
        np.random.shuffle(d['y_test'])

    if lhe:
        d['x_train'] = list(map(local_histogram_equalization, d['x_train'][:, 0, :, :].astype(np.uint8)))
        d['x_train'] = np.array(d['x_train'])
        d['x_train'] = d['x_train'].reshape(d['x_train'].shape[0], 1, 32, 32)
        d['x_train'] = d['x_train'].astype(np.float32)
        d['x_validation'] = list(map(local_histogram_equalization, d['x_validation'][:, 0, :, :].astype(np.uint8)))
        d['x_validation'] = np.array(d['x_validation'])
        d['x_validation'] = d['x_validation'].reshape(d['x_validation'].shape[0], 1, 32, 32)
        d['x_validation'] = d['x_validation'].astype(np.float32)
        d['x_test'] = list(map(local_histogram_equalization, d['x_test'][:, 0, :, :].astype(np.uint8)))
        d['x_test'] = np.array(d['x_test'])
        d['x_test'] = d['x_test'].reshape(d['x_test'].shape[0], 1, 32, 32)
        d['x_test'] = d['x_test'].astype(np.float32)

    if norm_255:
        # Normalizing whole data by dividing /255.0
        d['x_train'] = d['x_train'].astype(np.float32) / 255.0
        d['x_validation'] /= 255.0
        d['x_test'] /= 255.0

        mean_image = np.mean(d['x_train'], axis=0)
        dictionary = {'mean_image_' + colour: mean_image}
        with open('mean_image_' + colour + '.pickle', 'wb') as f_mean_image:
            pickle.dump(dictionary, f_mean_image)

        std = np.std(d['x_train'], axis=0)
        dictionary = {'std_' + colour: std}
        with open('std_' + colour + '.pickle', 'wb') as f_std:
            pickle.dump(dictionary, f_std)

    # Applying Mean Normalization
    if mean_norm:
        with open('mean_image_' + colour + '.pickle', 'rb') as f:
            mean_image = pickle.load(f, encoding='latin1')

        d['x_train'] -= mean_image['mean_image_' + colour]
        d['x_validation'] -= mean_image['mean_image_' + colour]
        d['x_test'] -= mean_image['mean_image_' + colour]

    # Applying STD Normalization
    if std_norm:
        with open('std_' + colour + '.pickle', 'rb') as f:
            std = pickle.load(f, encoding='latin1')
        d['x_train'] /= std['std_' + colour]
        d['x_validation'] /= std['std_' + colour]
        d['x_test'] /= std['std_' + colour]

    if transpose:
        d['x_train'] = d['x_train'].transpose(0, 3, 1, 2)
        d['x_validation'] = d['x_validation'].transpose(0, 3, 1, 2)
        d['x_test'] = d['x_test'].transpose(0, 3, 1, 2)

    # Returning preprocessed data
    return d