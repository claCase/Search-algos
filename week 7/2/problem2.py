import pandas as pd
import numpy as np
import sys


def load_data(path):
    data = np.asarray(pd.read_csv(path).values)
    return data


def normalize_data(data):
    means = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    data = (data - means) / sd
    return data


class LinearRegression:
    def __init__(self, n_weights, lr):
        self.weights = np.zeros((n_weights, 1))
        self.lr = lr

    def get_weights(self):
        return self.weights

    def loss(self, inputs):
        predictions = self.predict(inputs)
        return np.sum((predictions - inputs) ** 2)

    def predict(self, inputs):
        predictions = inputs.dot(self.weights)
        return predictions

    def gradient_step(self, inputs, outputs):
        predictions = self.predict(inputs)
        print(predictions.shape)
        error = predictions.flatten() - outputs.flatten()
        '''
        err = (predictions.flatten() - outputs.flatten()).reshape(-1,1)*inputs
        grad = (1 / inputs.shape[0]) * np.sum(err, axis=0)[:, np.newaxis]
        '''
        grad = inputs.T.dot(error)[:, np.newaxis] * (1 / inputs.shape[0])
        print(grad)
        self.weights -= self.lr * grad


def write_output(output, output_path):
    output = np.asarray(output).reshape(-1, 5)
    df = pd.DataFrame(output)
    df.to_csv(output_path, index=False, index_label=False, header=False)


def main():
    input_file = sys.argv[1].lower()
    output_file = sys.argv[2].lower()
    data = load_data(input_file)
    X = data[:, :-1]
    X = normalize_data(X)
    ones = np.ones(data.shape[0])[:, np.newaxis]
    X = np.hstack((ones, X))
    y = data[:, -1]
    print(X)
    # quit()
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, .62]
    lines = []
    rn = 100
    for lr in alphas:
        print("LR %f" % lr)
        linear = LinearRegression(X.shape[1], lr)
        i = 1
        if lr == .62:
            rn = 120
        for i in range(rn):
            i += 1
            linear.gradient_step(X, y)
            err = linear.loss(X)
            print(err)
        lines.append([lr, i, *linear.get_weights().flatten()])

    print(lines)
    write_output(lines, output_file)


if __name__ == '__main__':
    main()
