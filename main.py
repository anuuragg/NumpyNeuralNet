import numpy as np
import pandas as pd
import functions as fn

bias = 0.0
l_rate = 0.01
epochs = 10
epoch_loss = []

data, weights = fn.generate_data(50,3)

def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = fn.get_weighted_sum(feature, weights, bias)
            prediction = fn.sigmoid(w_sum)
            loss = fn.cross_entropy(target, prediction)

            individual_loss.append(loss)

            # gradient descent
            weights = fn.update_weights(weights, l_rate, target, prediction, feature)
            bias = fn.update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print('***************************')
        print("epoch", e)
        print(average_loss)

train_model(data, weights, bias, l_rate, epochs)

