import numpy as np
import copy

class LayerDense():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input = np.zeros(input_size)
        self.weight = np.random.rand(output_size, input_size)
        self.bias = np.ones(output_size)

    def forward(self, activation):
        self.output = np.zeros(self.output_size)
        for i in range (self.output_size):
            for j in range (self.input_size):
                self.output[i] += self.input[j] * self.weight[i, j]
            self.output[i] += self.bias[i]
        
        if (activation == "Sigmoid"):
            Sigmoid(self)
        elif (activation == "RELU"):
            RELU(self)

        return self.output

    def mutate(self):

        # self.weight = np.random.rand(self.output_size, self.input_size)
        # self.bias = np.zeros(self.output_size)

        change = 0.5
        rate = change * 2
        

        for i in range(len(self.weight)):
            for j in range(len(self.weight[0])):
                self.weight[i][j] *= (np.random.rand()*rate + 1 - change)
        for i in range (len(self.bias)):
            self.bias[i] *= (np.random.rand()*rate + 1 - change)

    def to_string(self):
        message = self.input
        return message
    
class Model():
    def __init__(self):
        self.layer = []

    def addLayer(self, x, y):
        self.layer.append(LayerDense(x,y))

    def mutate(self):
        for layer in self.layer:
            layer.mutate()

    def forward(self, input):
        self.layer[0].input = input
        for i in range(len(self.layer)-1):
            self.layer[i+1].input = self.layer[i].forward("RELU")
        output = self.layer[-1].forward("RELU")
        return output
      
    def to_string(self):
        message = ""
        for layer in self.layer:
            message += layer.to_string() + "\n"
        return message

def RELU(layer):
    for i in range (len(layer.output)):
        if(layer.output[i] < 0):
            layer.output[i] = 0

def Sigmoid(layer):
    for i in range (len(layer.output)):
        x = layer.output[i]
        layer.output[i] = (np.e**x)/(1+np.e**x)
    

def Cost(x, y):
    return abs(x-y)

# assign inputs and target outputs
inputs = [[1],[2],[3],[4],[5],[6],[10],[30],[50],[100]]
y = [[5],[7.5],[10],[12.5],[15],[17.5],[27.5],[77.5],[127.5],[252.5]]

# inputs = [[0,0],[0,1],[1,0],[1,1]]
# y = [[0],[0],[0],[1]]

model = Model()

model.addLayer(1,5)
model.addLayer(5,5)
model.addLayer(5,5)
model.addLayer(5,1)

best_model = Model()
best_model = copy.deepcopy(model)

best_cost = 0

for i in range (len(inputs)):

    output = model.forward(inputs[i])

    best_cost += Cost(output[0], y[i][0])

best_cost /= len(inputs)

print(output)
print(best_cost)


for i in range (5000):

    model.mutate()
    cost = 0

    for i in range (len(inputs)):

        output = model.forward(inputs[i])

        cost += Cost(output[0], y[i][0])

    cost /= len(inputs)

    if (cost < best_cost):
        best_cost = cost
        best_model = copy.deepcopy(model)
    else:
        model = copy.deepcopy(best_model)

    # print(output)
    # print(cost)

print()
print(best_model.forward(inputs[0]))
print(str(best_cost) + "\n")

print(best_model.forward([7]))




n = ""
while (True):
    print("Input: ")
    n = input()
    if (n == 'exit'):
        break
    print(best_model.forward([int(n)]))
