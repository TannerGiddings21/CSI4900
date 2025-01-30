import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean, variance
from functools import lru_cache
from math import sqrt
from typing import List, Tuple
import itertools
import json
import os
import time
from tqdm import tqdm

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Initializing parameters
input_shape = (28, 28, 1)
#batch_size = 64
#num_classes = 10
#epochs = 5

#Cleaning data
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

def make_model(
    cnn_layer_sizes : List[Tuple[int, Tuple[int, int]]],
    dense_layer_sizes : List[int]
):
    activation_function = 'relu'
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(input_shape))
    for i in range(len(cnn_layer_sizes)-1):
        model.add(tf.keras.layers.Conv2D(cnn_layer_sizes[i][0], cnn_layer_sizes[i][1], activation=activation_function))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if len(cnn_layer_sizes) > 0:
        model.add(tf.keras.layers.Conv2D(cnn_layer_sizes[-1][0], cnn_layer_sizes[-1][1], activation=activation_function))
    model.add(tf.keras.layers.Flatten())
    for elem in dense_layer_sizes:
        model.add(tf.keras.layers.Dense(elem, activation = activation_function))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.modelel.stop_training = True

"""callbacks = myCallback()

model = make_model([(10, (5,5)), (5, (3,3))], [64, 64], 'tanh')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=5,
                    validation_split=0.1,
                    callbacks=[callbacks])"""

#CONFIGURATIONS
cnn_layer_depth = [0,1]
cnn_layer_dimensions = [(3,3), (4,4), (5,5)]
num_cnn_layers = [1,2,3]
num_dense_layers = [1,2,3]
dense_layer_sizes = [64, 128, 256]
batch_size = 64

def is_decreasing(L : List[int]):
    prev = L[0]
    for i in range(1, len(L)):
        if L[i] < prev:
            return False
    return True

def gen_cnn_layer_combinations():
    L = []
    for layer in num_cnn_layers:
        for dim in cnn_layer_dimensions:
            L.append((layer, dim))
    return L
    
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.modelel.stop_training = True

def iterate_combinations():
    for cnn_depth in cnn_layer_depth:
        for cnn_layer_dim in itertools.product(gen_cnn_layer_combinations(), repeat=cnn_depth):
            for dense_depth in num_dense_layers:
                for dense_layers in itertools.product(dense_layer_sizes, repeat=dense_depth):
                    if is_decreasing(dense_layers):
                        callbacks = myCallback()
                        model = make_model(list(cnn_layer_dim), list(dense_layers))
                        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=4,
                            validation_split=0.1,
                            callbacks=[callbacks])
                        
                        yield model, str([cnn_layer_dim]), str([dense_layers]), history.history['acc'][-1], True
                    else:
                        yield tf.keras.models.Sequential(), "", "", 0, False
                    
    return 

#iterate_combinations()

def create_random_screen(eps = 0.01):
    return (np.random.rand(28,28) * 2 * eps) - eps

def selection(scores):
    #https://www.geeksforgeeks.org/python-indices-of-n-largest-elements-in-list/
    return sorted(range(len(scores)), key = lambda sub: scores[sub])[-25:]

def make_modifications(screen, randomness=0.001, eps=0.01):
    new_screen = screen + (np.random.rand(28,28) - 0.5) * 2 * randomness
    for i in range(28):
        for j in range(28):
            if new_screen[i][j] > eps:
                new_screen[i][j] = eps
            elif new_screen[i][j] < -eps:
                new_screen[i][j] = -eps
    return new_screen

def random_indices(n, total):
    L = [True] * n + [False] * (total - n)
    random.shuffle(L)
    return L

def find_max_index(L, scores):
    max_index = L[0]
    max_val = scores[0]
    for i in range(1, len(L)):
        if scores[i] > max_val:
            max_index = L[i]
            max_val = scores[i]
    return max_index, max_val

def sample(n, screens):
    index = random_indices(n, len(screens))
    return np.array([screens[i] for i in range(len(screens)) if index[i]])

def find_mean_var(m):
    l = m.reshape(784)
    return mean(l), variance(l)

@lru_cache()
def get_these_values(from_value):
    return np.array([x_train[i] for i in range(len(x_train)) if y_train[i][from_value] == 1])

def mean_difference_initialization(target, from_value, eps):
    x_target = get_these_values(target)
    x_from_value = get_these_values(from_value)
    cur = np.zeros(28,28)
    for _ in range(100):
        cur += x_from_value[random.randint(0, len(x_from_value))] - x_target[random.randint(0, len(x_target))]
    mean, var = find_mean_var(cur)
    return (cur - mean) / sqrt(var) * eps

def genetic_algorithm(model, pbar, eps=0.5, iterations=100, batch_size=100, randomness=0.05, target=7, from_value=1, init='rand'):
    if init=='rand':
        screens = [create_random_screen(eps=eps) for _ in range(batch_size)]
    else:
        screens = [mean_difference_initialization(target, from_value, eps) for _ in range(batch_size)]

    #num_samples = 1280
    x_train_no_target = get_these_values(from_value)
    y_sample = tf.one_hot(np.array(([target] * len(x_train_no_target))), depth=10)
    for i in range(iterations):
        scores = [model.evaluate(np.array([elem + s.reshape(28,28,1) for elem in x_train_no_target]), y_sample, verbose=0)[1] for s in screens]
        pbar.update(1)
        pbar.set_description(f"{from_value} -> {target} - Max: {round(max(scores), 2)}, Mean: {round(mean(scores), 2)}, Min: {round(min(scores), 2)}")
        
        if i == iterations - 1 or max(scores) == 1:
            return find_max_index(screens, scores)
        
        index = selection(scores)
        new_screens = []
        for elem in index:
            new_screens += [make_modifications(screens[elem], randomness=randomness, eps=eps) for i in range(2)]
        screens = new_screens
        del new_screens
    return find_max_index(screens, scores)

def get_two_diff_numbers(number_pairs):
    num1 = random.randint(0,9)
    num2 = random.randint(0,9)
    
    while num1 == num2 or [num1, num2] in number_pairs:
        num1 = random.randint(0,9)
        num2 = random.randint(0,9)
    return num1, num2

def send_to_github(model_counter, screen, score, number_counter):
    if os.path.exists(f'screens/model{model_counter}'):
        os.system(f'rm -r screens/model{model_counter}')
    os.mkdir(f'screens/model{model_counter}')
    np.savetxt(f'screens/model{model_counter}/screen_{number_counter}.txt', screen)
    with open(f'screens/model{model_counter}/scores{number_counter}.txt', 'w') as file:
        file.write(str(score))
    #os.system(f"git add screens/model{model_counter}")
    #os.system("git add model_info.json")
    #os.system("git add tracker.txt")
    #os.system(f"git commit -m \"Adding model{model_counter} info\"")
    #os.system("git push origin main")
    #os.system(f'rm -r screens/model{model_counter}')
    return

def run_project():
    try:
        with open('model_info.json', 'r') as file:
            models_info = json.load(file)
    except FileNotFoundError:
        models_info = {
            "CNN" : [],
            "Dense" : [],
            "accuracy" : [],
            "num1" : [],
            "num2" : [],
            "score" : [],
            "counter" : [],
        }
        
    #try:
    #    with open('tracker.txt', 'r') as file:
    #        starter = int(file.read())
    #except:
    #    starter = -1
        
    counter = 0
    num_iterations=100
    pbar2 = tqdm(total=num_iterations, desc="Attacking model")
        
    for model, cnn_layer_dim, dense_layers, acc, conti in iterate_combinations():
        number_pairs = []
        for i in range(5):
            if counter > 132 and conti:
                num1, num2 = get_two_diff_numbers(number_pairs)
                number_pairs.append([num1,num2])
                models_info['CNN'].append(cnn_layer_dim)
                models_info['Dense'].append(dense_layers)
                models_info['accuracy'].append(acc)
                models_info['num1'].append(num1)
                models_info['num2'].append(num2)

                pbar2.reset()
                screen, score = genetic_algorithm(model, pbar2, eps=0.5, iterations=num_iterations, batch_size=50, target=num1, from_value=num2)
                models_info['score'].append(score)
                models_info['counter'].append(counter)

                with open('tracker.txt', 'w') as file:
                    file.write(str(counter))
                
                with open('model_info.json', 'w') as file:
                    json.dump(models_info, file, indent=4)
                
                send_to_github(counter, screen, score, counter)
            print(f"Iteration {i}")
            counter += 1
    return
run_project()