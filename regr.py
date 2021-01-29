import numpy as np 

class predictor:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.rate = 0.0005
    
    def train(self, train_input, train_output):
        predicted_output = self.slope * train_input + self.intercept
        error_slope = 2*np.sum(np.dot((predicted_output - train_output), train_input)) / len(train_input)
        error_intercept = 2*np.sum(predicted_output - train_output) / len(train_input)
        self.slope = self.slope - np.dot(self.rate, error_slope)
        self.intercept = self.intercept - np.dot(self.rate, error_intercept)

    def predict(self, input):
        return self.slope * input + self.intercept


train_input = np.array([0, 5, 10.5, 15, 20, 25, 30, 40])
train_output = np.array([10, 13, 15, 17, 20, 22.5, 25, 30])

my_predictor = predictor()

for i in range(10000):
    my_predictor.train(train_input, train_output)

print(my_predictor.predict(16))
print(my_predictor.predict(35))

