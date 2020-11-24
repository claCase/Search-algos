import numpy as np
import sys
import pandas as pd


class Perceptron():
    def __init__(self, n_units=3, n_outputs=1, lr=1):
        self.n_outputs = n_outputs
        self.n_units = n_units
        self.weights = np.zeros(self.n_units * self.n_outputs).reshape(self.n_units, self.n_outputs)
        self.lr = lr

    def predict(self, input):
        output = input.dot(self.weights)
        if output > 0:
            return 1
        else:
            return -1

    def error(self, inputs, outputs):
        err = 0
        for i, o in zip(inputs, outputs):
            if self.predict(i) != o:
                err += 1
        return err

    def train(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            model_output = self.predict(input)
            if model_output * output <= 0:
                input = input.reshape(-1, 1)
                if model_output > output:
                    self.weights -= input
                else:
                    self.weights += input
        #print(self.weights)
        return self.weights, self.error(inputs, outputs)


def write_output(output, output_path):
    output = np.asarray(output).reshape(-1, 3)
    df = pd.DataFrame(output)
    df.to_csv(output_path, index=False, index_label=False, header=False)


def main():
    input_file = sys.argv[1].lower()
    output_file = sys.argv[2].lower()
    data = np.asarray(pd.read_csv(input_file, header=None).values)
    inputs = data[:, :2].reshape(-1, 2)
    last_col = np.ones(inputs.shape[0])[:, np.newaxis]
    inputs = np.hstack((inputs, last_col))
    outputs = data[:, -1].reshape(-1, 1)
    perceptron = Perceptron()
    err = perceptron.error(inputs, outputs)
    weights = []
    i = 0
    while err:
        #print(i)
        i += 1
        weight, err = perceptron.train(inputs, outputs)
        weights.append(weight.flatten())
    print(weights)
    write_output(weights, output_file)


if __name__ == "__main__":
    main()
