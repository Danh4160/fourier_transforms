import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import PIL
from matplotlib.colors import LogNorm


img = plt.imread('images\moonlanding.png').astype(float)
 
globalM = 0
globalN = 0


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


def fft_inverse(vector):
    # Divide and conquer
    N = len(vector)
    if N == 1:
        x = dft_inverse(vector)
        return x
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft(vector_even)
        vector_odd_trans = fft(vector_odd)

        k = np.arange(N)
        exp = (1j * 2 * np.pi / N) * k
        c = np.exp(exp)
        
        l_even = (1/N) * (vector_even_trans+c[:int(N/2)]*vector_odd_trans)
        l_odd = (1/N) * (vector_even_trans+c[int(N/2):]*vector_odd_trans)
        l = np.concatenate([l_even, l_odd])
        return l


def fft(vector):
    # Divide and conquer
    N = len(vector)
    if N == 1:
        x = dft(vector)
        return x
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft(vector_even)
        vector_odd_trans = fft(vector_odd)

        k = np.arange(N)
        exp = -1j * 2 * np.pi * k / N
        c = np.exp(exp)
        
        l_even = vector_even_trans+c[:int(N/2)]*vector_odd_trans
        l_odd = vector_even_trans+c[int(N/2):]*vector_odd_trans
        l = np.concatenate([l_even, l_odd])
        return l
    
    
def twoDftNormal(img, mode=dft):
    imgvector = np.asarray(img)
    imgvectorshape = imgvector.shape
    M = imgvectorshape[0]
    N = imgvectorshape[1]

    globalM = M
    globalN = N
    
    if mode == fft:
        # Need padding
        mpower = findNextPowerOf2(M)
        npower = findNextPowerOf2(N)
        mPadd = mpower - M
        nPadd = npower - N
        N += nPadd
        M += mPadd
        imgvector = np.pad(imgvector, pad_width=((0, mPadd), (0, nPadd)))
    

    resultVector = np.zeros((M, N), dtype=complex)

    for m in range(0,M):
        print(m)
        resultVector[m] = mode(imgvector[m])
        print(resultVector[m])

    for n in range(0,N):
        print(n)
        resultVector[:,n] = mode(resultVector[:,n])
        print(resultVector[:,n])
    

    print(f"image vector shape {imgvector.shape}")
    print(f"M {M}, N {N}")
    return (1/N*M) * resultVector if mode == fft_inverse else  resultVector
   

# Compute power of two greater than or equal to `n`
def findNextPowerOf2(n):
 
    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
 
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1       # unset rightmost bit
 
    # `n` is now a power of two (less than `n`)
 
    # return next power of 2
    return n << 1


def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed
    print("First mode")
    img2 = twoDftNormal(img)
   
    plt.figure()
    plt.imshow( np.abs(img2), norm =LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()

    # Reconstruct image 
    reconstructed_image = twoDftNormal(img2, dft_inverse)
    plt.figure()
    plt.imshow(reconstructed_image, plt.cm.gray)
    plt.title('Reconstructed Image')
    plt.show()

    return None

def second_mode():
    # Denoise an image by applying an FFT
    print("Second mode")
    # X = [1, 2, 3, 4, 5, 6, 7, 8]
    # print(fft(X))
    # print(np.fft.fft(X))
    # print(FFT(X))
    # Use FFT
    fft_image = twoDftNormal(img, fft)
   
    plt.figure()
    plt.imshow( np.abs(fft_image), norm =LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()

    print(f"Before filtering {fft_image}")
    # Filter out high frequencies
    fft_image[fft_image > (1 + np.exp(4j))] = 0
    filtered = fft_image
    print(f"After filtering {filtered}")
    # Change back to image
    reconstructed_image = twoDftNormal(filtered, fft_inverse).real
    plt.figure()
    plt.imshow(reconstructed_image, plt.cm.gray)
    plt.title('Reconstructed Image')
    plt.show()
    

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



