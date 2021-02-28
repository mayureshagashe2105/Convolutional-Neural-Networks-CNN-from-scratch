import numpy as np
import scipy.special
from tensorflow.keras.datasets import cifar10


class ConvolutionLayer:
    def __init__(self, n_filters, filter_size, stride_length, n_channels, input, learning_rate=0.001):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_channels = n_channels
        self.filters = np.random.randn(self.n_filters, self.n_channels, self.filter_size, self.filter_size)
        self.stride = stride_length
        self.output_dim = (input.shape[1] - self.filter_size) // self.stride + 1
        self.output_matrix = np.zeros((self.n_filters, self.output_dim, self.output_dim))
        self.lr = learning_rate
        self.input = input

    def feedforward(self, input):
        j, k, x, y = 0, 0, 0, 0
        for i in range(self.n_filters):
            while input.shape[1] - j >= self.filter_size:
                while input.shape[2] - k >= self.filter_size:
                    convoluted_values = []
                    for l in range(self.n_channels):
                        filtered_matrix = input[l, j:self.filter_size + j, k:self.filter_size + k]
                        temp_matrix = self.filters[i, l, :, :] * filtered_matrix
                        convoluted_values.append(np.sum(temp_matrix))

                    self.output_matrix[i, x, y] = sum(convoluted_values)
                    if y == self.output_dim - 1:
                        x += 1
                        y = 0
                    else:
                        y += 1

                    k += self.stride
                k = 0
                j += self.stride
            j = 0
            x, y = 0, 0

    def backpropagation(self, dl_dout):
        dl_df = np.zeros(self.filters.shape)
        j, k, x, y = 0, 0, 0, 0
        for i in range(self.n_filters):
            while self.input.shape[1] - j >= self.filter_size:
                while self.input.shape[2] - k >= self.filter_size:
                    stacking = np.zeros((1, self.filter_size, self.filter_size))
                    for l in range(self.n_channels):
                        filtered_matrix = self.input[l, j:self.filter_size + j, k:self.filter_size + k]
                        stacking = np.append(stacking, np.expand_dims(filtered_matrix, axis=0), axis=0)
                    stacking = np.delete(stacking, 0, axis=0)
                    dl_df[i] += stacking * dl_dout[i, j, k]
                    k += self.stride
                k = 0
                j += self.stride
            j = 0

        self.filters -= self.lr * dl_df


class MaxPoolingLayer:
    def __init__(self, mask_size, inputs):
        self.mask_size = mask_size
        self.inputs = inputs
        self.output_dim = inputs.shape[1] // self.mask_size
        self.output_matrix = np.zeros((self.inputs.shape[0], self.output_dim, self.output_dim))
        self.max_val_ind = []

    def feedforward(self, inputs):
        self.inputs = inputs
        j, k, x, y = 0, 0, 0, 0
        while self.inputs.shape[1] - j >= self.mask_size:
            while self.inputs.shape[2] - k >= self.mask_size:
                for l in range(self.inputs.shape[0]):
                    filtered_matrix = self.inputs[l, j:self.mask_size + j, k:self.mask_size + k]
                    self.output_matrix[l, x, y] = np.max(filtered_matrix)
                    a = np.where(filtered_matrix == np.max(filtered_matrix))
                    self.max_val_ind.append((l, a[0][0], a[1][0]))
                if y == self.output_dim - 1:
                    y = 0
                    x += 1
                else:
                    y += 1
                k += self.mask_size
            k = 0
            j += self.mask_size

    def backpropogation(self, grad):
        j, k = 0, 0
        arr = np.zeros(self.inputs.shape)
        for i in self.max_val_ind:
            while j != grad.shape[1] - 1:
                while k != grad.shape[2] - 1:
                    for l in range(grad.shape[0]):
                        arr[i] = grad[l, j, k]
                    k += 1
                k = 0
                j += 1


        self.dl_dout = arr


class DenseLayerNetwork:
    def __init__(self, input_nodes, output_nodes, learning_rate=0.001):
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.weights = np.random.randn(self.onodes, self.inodes) / 255
        self.lr = learning_rate
        self.activation_function_Softmax = lambda x: scipy.special.softmax(x)

    def feedforward(self, input_img):
        self.original_img = input_img
        self.input_img = input_img
        self.input_img = input_img.reshape(-1, 1)
        self.outputs = np.dot(self.weights, self.input_img)
        self.outputs = self.activation_function_Softmax(self.outputs)

    def backpropogation(self, dl_dout):
        for i in range(len(dl_dout)):
            outputs = np.dot(self.weights, self.input_img)
            outputs_exp = np.exp(outputs)
            S_total = np.sum(outputs_exp)

            dy_dz = -outputs_exp[i] * outputs_exp / (S_total ** 2)
            dy_dz[i] = outputs_exp[i] * (S_total - outputs_exp[i]) / (S_total ** 2)

            dz_dw = self.input_img
            dz_dinp = self.weights

            dl_dz = i * dy_dz

            dl_dw = (dz_dw.dot(dl_dz.transpose())).transpose()
            dL_dinp = dz_dinp.transpose().dot(dl_dz)

            self.weights -= self.lr * dl_dw
        return dL_dinp.reshape(self.original_img.shape)


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
X_train = X_train.reshape(-1, 3, 32, 32)
print(np.product(X_train[0].shape))


def train_CNN(X_train, y_train, X_test, y_test, epochs=1):
    conv1 = ConvolutionLayer(32, 3, 1, 3, X_train[0])
    pool1 = MaxPoolingLayer(3, conv1.output_matrix)
    inputs_to_dense = np.product(pool1.output_matrix.shape)
    dense1 = DenseLayerNetwork(inputs_to_dense, 10)
    for j in range(epochs):
        for i in range(len(X_train)):
            conv1.feedforward(X_train[i] / 255)
            pool1.feedforward(conv1.output_matrix)
            dense1.feedforward(pool1.output_matrix)

            grad = np.zeros(10)
            grad[y_train[i]] = -1 / dense1.outputs[y_train[i]]

            grad = dense1.backpropogation(grad)
            pool1.backpropogation(grad)
            conv1.backpropagation(pool1.dl_dout)

            print(i)
        print(j)

    for i in range(len(y_test)):
        conv1.feedforward(X_test[i] / 255)
        pool1.feedforward(conv1.output_matrix)
        dense1.feedforward(pool1.output_matrix)
        print(np.argmax(dense1.outputs), y_test[i])


train_CNN(X_train, y_train, X_test, y_test, epochs=2)

