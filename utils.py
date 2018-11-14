import numpy as np
import struct
import matplotlib.pyplot as plt

def parse_images(filename):
    f = open(filename, "rb")
    magic, size = struct.unpack('>ii', f.read(8))
    sx, sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im = struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/256.0 for x in im])
    return np.array(X)


def parse_labels(filename):
    f = open(filename, "rb")
    magic, size = struct.unpack('>ii', f.read(8))
    y = []
    for i in range(size):
        y.append(struct.unpack('B', f.read(1))[0])
    return np.array(y)


def plot_image(X):
    row_sqrt = np.sqrt(X.shape[0]).astype(int)
    plt.axis('off')
    plt.imshow(X.reshape(row_sqrt, row_sqrt), cmap='gray')
    plt.show()


def plot_images(X, rows, cols):
    row_sqrt = np.sqrt(X.shape[1]).astype(int)
    _, arrs = plt.subplots(rows, cols)

    def make_plot(row):
        if rows > 1 and cols > 1:
            plot = arrs[row / cols, row % cols]
        else:
            plot = arrs[row]
        plot.axis('off')
        return plot.imshow(X[row, :].reshape(row_sqrt, row_sqrt), cmap='gray')

    for row in range(rows*cols):
        make_plot(row)

    plt.show()
