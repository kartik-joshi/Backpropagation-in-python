#Program for backpropogation for  6 class Dermatology dataset


from math import exp
from random import seed
from random import random
from csv import reader
from numpy import matrix

print('program execution start')


# here cost matrix is initialized for misclassification
CostMatrix = matrix([[0,0.3,0.6,0.9,1.2,1.5],[0.3,0,0.3,0.6,0.9,1.2],[0.6,0.3,0,0.3,0.6,0.9],[0.9,0.6,0.3,0,0.3,0.6],[1.2,0.9,0.6,0.3,0,0.3],[1.5,1.2,0.9,0.6,0.3,0]])
AysCostMatrix = matrix([[0,0.6,0.3,0.6,0.9,1.5],[0.3,0,0.9,1.2,1.5,0.3],[0.5,0.1,0,1.3,0.7,1],[0.3,0.9,1.1,0,1.4,0.3],[1.5,0.3,0.7,1.2,0,0.4],[0.5,1.2,0.4,0.9,0.4,0]])


#when we dont want to apply cost
NoCostMatrix = matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
NoCostMatrix1 = matrix([[0,1,1,1,1,1],[1,0,1,1,1,1],[1,1,0,1,1,1],[1,1,1,0,1,1],[1,1,1,1,0,1],[1,1,1,1,1,0]])
ConfusionMatrix1 = matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
ConfusionMatrix2 = matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
ConfusionMatrix3 = matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

print("Symmetric Cost Matrix for Dermatology Dataset : ")
i = 0
for letter in 'pyhton':
    print(CostMatrix[i])
    i = i + 1

print("ASymmetric Cost Matrix for Dermatology Dataset : ")
i = 0
for letter in 'pyhton':
    print(AysCostMatrix[i])
    i = i + 1


# Read datasets from CSV input file
def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string columns to float in input dataset
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer in input dataset (last column with class value)
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


#training dataset
traindataset = Read_file('traindata.csv')
#testdataset
testdataset = Read_file('testdata.csv')


# change string column values to float
for i in range(len(traindataset[0]) - 1):
    str_column_to_float(traindataset, i)
# convert last column to integers
str_column_to_int(traindataset, len(traindataset[0]) - 1)
# change string column values to float
for i in range(len(testdataset[0]) - 1):
    str_column_to_float(testdataset, i)
# convert last column to integers
str_column_to_int(testdataset, len(testdataset[0]) - 1)

#normalize both training and test dataset to get better result
minmax = dataset_minmax(traindataset)
normalize_dataset(traindataset, minmax)
minmax = dataset_minmax(testdataset)
normalize_dataset(testdataset, minmax)


# Initialize a network
def init_nw(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate,difference):
    if(difference >= 1):
        new_l_rate = l_rate + (0.1 * difference)
    else: new_l_rate = l_rate
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += new_l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += new_l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network_withcost(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            difference = 0
            outputs = forward_propagate(network, row)
            difference = CostMatrix[outputs.index(max(outputs)),row[-1]]
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate,difference)
    print('No of Iterations =%d,  Error at last iteration=%.3f' % (epoch, sum_error))

# Train a network for a fixed number of epochs with Asymmetric cost matrix
def train_network_withcostA(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            difference = 0
            outputs = forward_propagate(network, row)
            difference = AysCostMatrix[outputs.index(max(outputs)),row[-1]]
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate,difference)
    print('No of Iterations =%d,  Error at last iteration=%.3f' % (epoch, sum_error))

# Train a network for a fixed number of epochs
def train_network_withoutcost(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            difference = 0
            outputs = forward_propagate(network, row)
            difference = NoCostMatrix1[outputs.index(max(outputs)), row[-1]]
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate, difference)
    print('No of Iterations =%d,  Error at last iteration=%.3f' % (epoch, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

n_hidden = 15
n_inputs = len(traindataset[0]) - 1
n_outputs = len(set([row[-1] for row in traindataset]))
network1 = init_nw(n_inputs, n_hidden, n_outputs)
network2 = network1
network3 = network1
print('Training Backprop aglo with considering Misclassification Cost (symmetric cost matrix)')
train_network_withcost(network1, traindataset, 0.5, 500, n_outputs)
total = 0
misclassification = 0
total_missclassification_cost = 0.0
for row in testdataset:
    total = total + 1
    prediction = predict(network1, row)
    if (row[-1] != prediction):
        ConfusionMatrix1[row[-1], prediction] = ConfusionMatrix1[row[-1], prediction] + 1
        misclassification = misclassification + 1
        total_missclassification_cost = total_missclassification_cost + CostMatrix[row[-1], prediction]
Accuracy =(total - misclassification)*100/total
print('Misclassified :=%d, out of:%d' % (misclassification,total))
print('Total Misclassification cost :=%f ' % (total_missclassification_cost))
print('Accuracy:=%.3f' %(Accuracy))
print("Confusion matrix: ")
i = 0
for letter in 'pyhton':
    print(ConfusionMatrix1[i])
    i = i + 1

# network2 = init_nw(n_inputs, n_hidden, n_outputs)

print('Training Backprop algo with considering Misclassification Cost (asymmetric cost matrix)')
train_network_withcostA(network2, traindataset, 0.5, 500, n_outputs)
total = 0
misclassification = 0
total_missclassification_cost = 0.0
for row in testdataset:
    total = total + 1
    prediction = predict(network2, row)
    if(row[-1] != prediction):
        ConfusionMatrix2[row[-1],prediction] = ConfusionMatrix2[row[-1],prediction]  + 1
        misclassification = misclassification + 1
        total_missclassification_cost = total_missclassification_cost + AysCostMatrix[row[-1],prediction]
Accuracy =(total - misclassification)*100/total
print('Misclassified :=%d, out of:%d' % (misclassification,total))
print('Total Misclassification cost :=%f ' % (total_missclassification_cost))
print('Accuracy:=%.3f' %(Accuracy))
print("Confusion matrix: ")
i = 0
for letter in 'pyhton':
    print(ConfusionMatrix2[i])
    i = i + 1




# network3 = init_nw(n_inputs, n_hidden, n_outputs)
print('Training with normal backpropogation without considering Misclassification Cost')
train_network_withoutcost(network3, traindataset, 0.5, 500, n_outputs)
total = 0
misclassification = 0
total_missclassification_cost1 = 0.0
total_missclassification_cost2 = 0.0
total_missclassification_cost3 = 0.0
for row in testdataset:
    total = total + 1
    prediction = predict(network3, row)
    if (row[-1] != prediction):
        misclassification = misclassification + 1
        ConfusionMatrix3[row[-1], prediction] = ConfusionMatrix3[row[-1], prediction] + 1
        total_missclassification_cost1 = total_missclassification_cost1 + NoCostMatrix1[row[-1], prediction]
        total_missclassification_cost2 = total_missclassification_cost2 + CostMatrix[row[-1], prediction]
        total_missclassification_cost3 = total_missclassification_cost3 + AysCostMatrix[row[-1],prediction]
Accuracy =(total - misclassification)*100/total
print('Misclassified :=%d, out of:%d' % (misclassification,total))
print('Total Misclassification  with symmetric cost :=%f ' % (total_missclassification_cost2))
print('Total Misclassification  with assymetric cost  :=%f ' % (total_missclassification_cost3))
print('Accuracy:=%.3f' %(Accuracy))
print("Confusion matrix: ")
i = 0
for letter in 'pyhton':
    print(ConfusionMatrix3[i])
    i = i + 1

print('program execution ends')