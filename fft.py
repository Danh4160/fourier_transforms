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


def fft_driver(threshold, vector):
    res = fft(threshold, vector, [])
    return res

def fft(threshold, vector, l):
    # Divide and conquer
    N = len(vector)
    if threshold == N:
        x = dft(vector)
        return x
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft(threshold, vector_even, l)
        vector_odd_trans = fft(threshold, vector_odd, l)

        k = np.arange(N)
        exp = (-1j * 2 * np.pi / N) * k
        c = np.exp(exp)
        
        l_even = vector_even_trans+c[:int(N/2)]*vector_odd_trans
        l_odd = vector_even_trans+c[int(N/2):]*vector_odd_trans
        l = np.concatenate([l, l_even, l_odd])
        return l

def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed
    print("First mode")
    # Testing
    # X = [1,2,3,4,5,6,7,0] 
    # dft_vector = dft(X)
    # dft_inverse_vector = dft_inverse(dft(X))
    # print(X)
    # print(dft_vector)
    # print(dft_inverse_vector)

    # Testing FFT
    X = [1,2,3,4,5,6,7,0] 
    fft_vector = fft_driver(1, X)
    print(fft_vector)
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



