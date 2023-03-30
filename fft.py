import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import PIL
from matplotlib.colors import LogNorm


img = plt.imread('images\moonlanding.png').astype(float)
originalM, originalN = np.asarray(img).shape

def dft(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    exp = (-1j * 2 * np.pi / N) * k * n
    c = np.exp(exp)
    return np.dot(vector, c)


def dft_inverse_fft(vector): 
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    exp = (1j * 2 * np.pi / (N/2)) * k * n
    c = np.exp(exp)
    return (1/N) * np.dot(vector, c)


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
        x = dft_inverse_fft(vector)
        # print(f"Called dft inverse, vector is {x}")
        return x
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft_inverse(vector_even)
        vector_odd_trans = fft_inverse(vector_odd)

        k = np.arange(N)
        exp = (1j * 2 * np.pi / N) * k
        c = np.exp(exp)
        # print(f"N {N} used")
        # print(f"vector even {vector_even_trans}")
        # print(f"vector odd {vector_odd_trans}")
        l_even = (vector_even_trans+c[:int(N/2)]*vector_odd_trans) 
        l_odd = (vector_even_trans+c[int(N/2):]*vector_odd_trans) 
        l = np.concatenate([l_even, l_odd])
        # print(f"Concatenated l {l}")
        return l / len(l_even)


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
        #print(m)
        resultVector[m] = mode(imgvector[m])
        #print(resultVector[m])

    for n in range(0,N):
        #print(n)
        resultVector[:,n] = mode(resultVector[:,n])
        #print(resultVector[:,n])
    

    #print(f"image vector shape {imgvector.shape}")
    #print(f"M {M}, N {N}")

    if mode == fft_inverse:
        resultVector = (1/N*M) * resultVector
        resultVector = resultVector[:originalM, :originalN]
        return resultVector
    return resultVector
   

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
    reconstructed_image = twoDftNormal(img2, dft_inverse).real
    plt.figure()
    plt.imshow(reconstructed_image, plt.cm.gray)
    plt.title('Reconstructed Image')
    plt.show()

    return None

def second_mode():
    # Denoise an image by applying an FFT
    print("Second mode")
    # X = [0, 10, 255, 21, 69, 420, 7, 1]
    # fft_image = fft(X)
    # print(f"Our FFT {fft_image}")
    # print(f"Their FFT {np.fft.fft(X)}")
    # print(f"Our Inverse FFT {fft_inverse(fft_image)}")
    # print(f"Their Inverse FFT {np.fft.ifft(fft_image)}")

    # Use FFT
    fft_image = twoDftNormal(img, fft)
   
    plt.figure()
    plt.imshow( np.abs(fft_image), norm =LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()

    # y = fft_image
    # x = np.arange(0, len(y), 1)
    # # Plot frequencies
    # time_step = 0.02
    # time_vec = np.arange(0, 20, time_step)
    # plt.figure(figsize=(6, 5))
    # plt.plot(x, fft_image, label='Original signal')
    # plt.show()

    # print(f"Before filtering {fft_image}")
    fft_image[(fft_image.real > (np.pi / 2)) & (fft_image.real < (3 * np.pi / 2))] = 0
    # fft_image[fft_image > (3 * np.pi / 2)] = 0
    # fft_image[fft_image < (np.pi / 2)] = 0
    # fft_image[fft_image < (3 * np.pi / 2)] = 0
    # filtered = fft_image
    # r, c = fft_image.shape
    # fft_image[int(r*0.1):int(r*(1-0.1))] = 0
    # # Similarly with the columns:
    # fft_image[:, int(c*0.1):int(c*(1-0.1))] = 0
    # print(f"After filtering {filtered}")
    # Change back to image
    reconstructed_image = twoDftNormal(fft_image, fft_inverse).real
    plt.figure()
    plt.imshow(reconstructed_image, plt.cm.gray)
    plt.title('Reconstructed Image')
    plt.show()
    

    return None

def third_mode():
    # Compressing and saving image
    print("Third mode")
    a = 2  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure()
    plt.suptitle("Compression of Image")
    arrayImg = np.asarray(img)
    print(arrayImg.shape)
    fftimg = twoDftNormal(img, fft)
    fftSorted = np.sort(np.abs(fftimg.reshape(-1)))
    for compressPerc in (1,0.8,0.6,0.4,0.2,0.05):
        thresh = fftSorted[int(np.floor((1-compressPerc) * len(fftSorted)))]
        ind = np.abs(fftimg)>thresh
        lowFarray = fftimg * ind
        lofArray = twoDftNormal(lowFarray, fft_inverse).real
        #plt.figure()
        numberOfNonzero = np.count_nonzero(lowFarray)
        savearray = np.asarray(lowFarray)

        sizelowarray = lowFarray.shape
        sizex = sizelowarray[0]
        sizey = sizelowarray[1]
        np.savetxt('data'+str(c)+'.csv',savearray,delimiter=',')
        plt.subplot(a,b,c)
        plt.imshow(lofArray, plt.cm.gray)
        plt.title('Compression =' + str(100 - compressPerc * 100) + '%')
        print('Number of nonzero Fourier coefficients for Compression = '+ str(100 - compressPerc * 100) + '%' + ' is equal to: ' + str(numberOfNonzero) + ' and Sparsity is equal to: ' + str(1-(numberOfNonzero/(sizex*sizey))))
        c=c+1
    plt.show()
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



