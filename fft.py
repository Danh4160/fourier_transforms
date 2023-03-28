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


def fft_inverse(threshold, vector, l):
    # Divide and conquer
    N = len(vector)
    if threshold == N:
        x = dft_inverse(vector)
        return x
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft(threshold, vector_even, l)
        vector_odd_trans = fft(threshold, vector_odd, l)

        k = np.arange(N)
        exp = (1j * 2 * np.pi / N) * k
        c = np.exp(exp)
        
        l_even = (1/N) * (vector_even_trans+c[:int(N/2)]*vector_odd_trans)
        l_odd = (1/N) * (vector_even_trans+c[int(N/2):]*vector_odd_trans)
        l = np.concatenate([l, l_even, l_odd])
        return l


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
    
def twoDftNormal(img):
    imgvector = np.asarray(img)
    imgvectorshape = imgvector.shape
    M = imgvectorshape[0]
    N = imgvectorshape[1]
    mpower = findNextPowerOf2(M)
    npower = findNextPowerOf2(N)
    
    
    mPadd = mpower - M
    nPadd = npower - N
    
    
    realM = M + mPadd
    realN = N + nPadd

    globalM = M
    globalN = N
   # imgvector = np.pad(imgvector, pad_width=((0, mPadd), (0, nPadd)))
   # resultVector = np.zeros((realM, realN), dtype=complex)

    #imgvector = np.pad(imgvector, pad_width=((0, mPadd), (0, nPadd)))
    resultVector = np.zeros((M, N), dtype=complex)

    #print(resultVector.shape)
    # print(imgvector[0])

    
    # for m in range(0,realM):
    #     print(imgvector.shape)
    #     print(resultVector.shape)
    #     print(m)
    #     resultVector[m] = dft(imgvector[m])
    #     print(resultVector[m])

    # for n in range(0,realN):
    #     print(n)
    #     resultVector[:,n] = dft(resultVector[:,N])
    #     print(resultVector[:,n])

    for n in range(0,N):
         print(n)
         resultVector[:,n] = dft(imgvector[:,n])
         print(resultVector[:,n])

    for m in range(0,M):
         print(m)
         resultVector[m] = dft(resultVector[m])
         print(resultVector[m])


    

    # for n in range(0,realN):
    #      print(n)
    #      resultVector[:,n] = dft(imgvector[:,n])
    #      print(resultVector[:,n])

    # for m in range(0,realM):
    #      print(m)
    #      resultVector[m] = dft(resultVector[m])
    #      print(resultVector[m])

    # for m in range(realM):
    #     resultVector[:m] = dft(resultVector[:m])
    #     print(resultVector[:m])

    return resultVector
   
 #Python program to find
#smallest power of 2
#greater than or equal to n
import math



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

    X = [1, 2, 3, 4]
    print(f"Array: {X}")
    # Testing DFT
    dft_vector = dft(X)
    dft_inverse_vector = dft_inverse(dft_vector)
    print(f"DFT result {dft_vector}")

    # Testing Inverse DFT
    print(f"Inverse DFT result {dft_inverse_vector}")

    # Testing FFT
    #X = [1,2,3,4,5,6,7,0] 
    #fft_vector = fft_driver(1, X)
    #print(fft_vector)
    img2 = twoDftNormal(img)
    # Output img with window name as 'image'
    #np.abs(im_fft), norm=LogNorm(vmin=5)
    #cv2.imshow('image', np.abs(img2), norm =LogNorm(vmin=5))
   
    plt.figure()
    plt.imshow( np.abs(img2), norm =LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return None

def second_mode():
    # Denoise an image by applying an FFT
    print("Second mode")

    im_vector = plt.imread()
    # Use FFT
    fft_vector = fft(1, )

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



