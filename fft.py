import sys
import numpy as np
import matplotlib as plt
import math




def dft(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    exp = (-1j * 2 * np.pi / N) * k * n
    c = np.exp(exp)
    return np.dot(vector, c)

def dft_inverse(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    exp = (1j * 2 * np.pi / N) * k * n
    c = np.exp(exp)
    return (1/N) * np.dot(vector, c)


def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed
    print("First mode")
    # Testing
    X = np.arange(3)
    dft_vector = dft(X)
    dft_inverse_vector = dft_inverse(dft(X))
    print(X)
    print(dft_vector)
    print(dft_inverse_vector)
    return None


def second_mode():
    # Denoise an image by applying an FFT
    print("Second mode")
    return None

def third_mode():
    # Compressing and saving image
    print("Third mode")
    return None

def fourth_mode():
    # Plotting runtime graphs for report
    print("Fourth mode")
    return None


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}:{arg}")
    
    mode = "1"
    image = "moonlanding.png"

    if len(sys.argv) - 1 == 2:
        mode = sys.argv[1]
        image = sys.argv[2]
    elif len(sys.argv) -1 == 1: 
        mode = sys.argv[1]
    
    print(f"mode is {mode}; image is {image}")


    if mode == "1":
        default_mode()

    elif mode == "2":
        second_mode()

    elif mode == "3":
        third_mode()

    else:
        fourth_mode()



