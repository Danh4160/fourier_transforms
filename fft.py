import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import cv2
import PIL
from matplotlib.colors import LogNorm
import time


img = plt.imread('images\moonlanding.png').astype(float)
originalM, originalN = np.asarray(img).shape

def dft(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    c = np.exp(-1j * 2 * np.pi * k * n / N)
    return np.dot(vector, c)


def dft_inverse_fft(vector): 
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    c = np.exp(1j * 2 * np.pi * k * n / N)
    return (1/N) * np.dot(vector, c)


def dft_inverse(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    c = np.exp(1j * 2 * np.pi * k * n / N)
    return (1/N) * np.dot(vector, c)


def fft_inverse(vector):
    # Divide and conquer
    N = vector.shape[0]
    if N <= 16:
        # print(f"Called dft inverse, vector is {x}")
        return dft_inverse(vector) * N
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft_inverse(vector_even)
        vector_odd_trans = fft_inverse(vector_odd)

        k = np.arange(N)
        c = np.exp(1j * 2 * np.pi * k / N)
        vector_even_trans = np.concatenate([vector_even_trans, vector_even_trans])
        vector_odd_trans = np.concatenate([vector_odd_trans, vector_odd_trans])           
        return (vector_even_trans + c * vector_odd_trans)                                                                                                                   


def fft(vector):
    # Divide and conquer
    N = len(vector)
    if N <= 16:
        return dft(vector)
    else:
        # Get the even and odd index numbers
        vector_even, vector_odd = vector[::2], vector[1::2]
        vector_even_trans = fft(vector_even)
        vector_odd_trans = fft(vector_odd)

        k = np.arange(N)
        c = np.exp(-1j * 2 * np.pi * k / N)
        
        vector_even_trans = np.concatenate([vector_even_trans, vector_even_trans])
        vector_odd_trans = np.concatenate([vector_odd_trans, vector_odd_trans])
        return vector_even_trans + c * vector_odd_trans


def twoDftNormal(img, test=False, mode=dft):
    imgvector = np.asarray(img)
    imgvectorshape = imgvector.shape
    M = imgvectorshape[0]
    N = imgvectorshape[1]
    print(M, N)
    
    if mode == fft:
        # Need padding
        print("pad")
        mpower = findNextPowerOf2(M)
        npower = findNextPowerOf2(N)
        mPadd = mpower - M
        nPadd = npower - N
        N += nPadd
        M += mPadd
        imgvector = np.pad(imgvector, pad_width=((0, mPadd), (0, nPadd)))

    resultVector = np.zeros((M, N), dtype=complex)
    
    for m in range(0,M):
        # print(m)
        resultVector[m] = mode(imgvector[m])
        # print(resultVector[m])


    # FFT on columns
    for n in range(0,N):
        # print(n)
        resultVector[:,n] = mode(resultVector[:,n])
        # print(resultVector[:,n])


    if mode == fft_inverse:
        if test:
            resultVector = (1/(N * M)) * resultVector[:M, :N]
            return resultVector
        else:
            resultVector = (1/(N * M)) * resultVector[:originalM, :originalN]
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

def test_runtime():
    sizes = [5,6,7,8,9,10]
    dft_final = []
    fft_final = []
    for size in sizes:
        arr = np.random.rand(2 ** size, 2 ** size)
        dft_time_res = []
        fft_time_res = []
        for i in range(10):
            # Testing DFT
            print(f"DFT Round {size}.{i}")
            start = time.time()
            twoDftNormal(arr, mode=dft)
            end = time.time()
            dft_time_res.append(end - start)

            # Testing FFT
            print(f"FFT Round {size}.{i}")
            start = time.time()
            twoDftNormal(arr, mode=fft)
            end = time.time()
            fft_time_res.append(end - start)

        dft_final.append(dft_time_res)
        fft_final.append(fft_time_res)

    dft_mean = np.mean(dft_final, axis=1)
    dft_std = np.std(dft_final, axis=1)
    fft_mean = np.mean(fft_final, axis=1)
    fft_std = np.std(fft_final, axis=1)

    print(f"DFT mean {dft_mean} std {dft_std}")
    print(f"FFT mean {fft_mean} std {fft_std}")

    for i, size in enumerate(sizes):
        print(f"Array with size 2^{size}")
        print(f"DFT Mean={dft_mean[i]}\tStd={dft_std[i]}")
        print(f"FFT Mean={fft_mean[i]}\tStd={fft_std[i]}")
        print("------------------------------------------------------")
    
    return dft_mean, dft_std, fft_mean, fft_std


def second_mode_test1(fft_input):
    # Remove high frequencies in the middle of the plot, keeping the corners only
    amount = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.50, 0.90]
    for t in amount:
        fft_image_copy = np.copy(fft_input)
        M, N = fft_image_copy.shape
        print(f"Shape of image: {fft_image_copy.shape}")
        fft_image_copy[int(M * t):, :int(M * (1-t))] = 0
        fft_image_copy[:,int(N * t):int(N * (1-t))] = 0

        reconstructed_image =  twoDftNormal(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent High Frequencies Removed: {}".format(100 * t),fontsize=22)
        plt.savefig(f'./mode_2_results/test1_{t}_with16threshold.png')
        # plt.show()
    plt.close()
    return None

def second_mode_test2(fft_input):
    # In each row, sort from high frequencies to low frequencies
    # Set to 0 the sorted array from the left meaning that we leave out high frequencies
    # Keep low frequencies, remove the top frequencies
    _, n = fft_input.shape
    amount = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.50, 0.90] 
    for t in amount:
        fft_image_copy = np.copy(fft_input)
        for row in fft_image_copy:
            index = np.argsort(np.abs(row))[::-1]
            row[index[:int(n * t)]] = 0

        reconstructed_image =  twoDftNormal(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent High Frequency Removed in each row: {}%".format(100*t),fontsize=22)
        plt.savefig(f'./mode_2_results/test2_{t}_with16threshold.png')
        # plt.show()
    plt.close()
    return None

def second_mode_test3(fft_input):
    # In each row, remove percent of low frequencies 
    _, n = fft_input.shape
    amount = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.50, 0.90] 
    for t in amount:
        fft_image_copy = np.copy(fft_input)
        for row in fft_image_copy:
            index = np.argsort(np.abs(row))
            row[index[:int(n * t)]] = 0
        reconstructed_image =  twoDftNormal(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent Low Frequency Removed in each row: {}%".format(100 * t),fontsize=22)
        plt.savefig(f'./mode_2_results/test3_{t}_with16threshold.png')
        # plt.show()
    plt.close()
    return None

def second_mode_test4(fft_input):
    # In each row, remove percent of low frequencies and high frequencies
    _, n = fft_input.shape
    amount = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.4, 0.47, 0.49] 
    for t in amount:
        fft_image_copy = np.copy(fft_input)
        np.transpose(fft_image_copy)
        for row in fft_image_copy:
            index = np.argsort(np.abs(row))
            row[index[:int(n * t)]] = 0
            row[index[-int(n * t):]] = 0

        reconstructed_image =  twoDftNormal(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent Low/High Frequency Removed in each row: {}%".format(100 * t),fontsize=22)
        plt.savefig(f'./mode_2_results/test4_{t}_with16threshold.png')
        # plt.show()
    plt.close()
    return None

def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed
    print("First mode")
    fft_image = twoDftNormal(img, mode=fft)
    
    plt.figure(figsize=(15,5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(fft_image.real), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("Fourier Transform"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Side by side comparison",fontsize=22)
    plt.savefig("./mode_1_results/originalvsft.png")
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
    
    # Use FFT
    fft_image = twoDftNormal(img, mode=fft)
   
    plt.figure()
    plt.imshow(np.abs(fft_image), norm =LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()

    # Since the frequencies in a fourier transform are index based, and due to its symmetry, we know 
    # that the highest frequencies would be in the neighborhood of matrix' heigth and width. 
    # Indeed, the we performed the transformation first on the row then on the columns.
    second_mode_test1(fft_image)

    second_mode_test2(fft_image)

    second_mode_test3(fft_image)

    second_mode_test4(fft_image)

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

    # confidence_level: 0.97  
    factor = 2

    dft_mean, dft_std, fft_mean, fft_std = test_runtime()
    dft_std = np.multiply(factor, dft_std)
    fft_std = np.multiply(factor, fft_std)
    size = ['32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024']

    # Plot 
    plt.plot(size, dft_mean, color='b', label='DFT')
    plt.plot(size, fft_mean, color='r', label='FFT')
    plt.errorbar(size, dft_mean, yerr=dft_std)
    plt.errorbar(size, fft_mean, yerr=fft_std)
    plt.title("Runtime DFT and FFT per size")
    plt.xlabel("Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("./mode_4_results/runtime.png")
    plt.show()
    return None

def test_correctness():
    # Test for differences using FFT
    fft_actual = twoDftNormal(img, mode=fft)
    fft_expected = np.fft.fft2(img, s=fft_actual.shape)
    difference = np.abs(fft_expected - fft_actual)
    s = np.sum(difference) / (img.shape[0] * img.shape[1])
    print("---------------- TEST 1 ----------------------")
    print(f"Correcness test 2D-FFT: Numpy vs Ours returns {np.allclose(fft_expected, fft_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences in inverse FFT
    test_array = np.random.rand(512, 512) 
    fft_inverse_expected = np.fft.ifft2(test_array)
    fft_inverse_actual = twoDftNormal(test_array, test=True, mode=fft_inverse)
    # fft_inverse_actual = inverse_dft2_fast(test_array)
    difference = np.abs(fft_inverse_expected - fft_inverse_actual)
    s = np.sum(difference) / (512 * 512)
    print("---------------- TEST 2 ----------------------")
    print(f"Correcness test 2D-FFT-Inverse: Numpy vs Ours custom returns {np.allclose(fft_inverse_expected, fft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences using DFT
    test_array = np.random.rand(512)
    dft_expected = np.fft.fft(test_array)
    dft_actual = dft(test_array)
    difference = np.abs(fft_expected - fft_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 3 ----------------------")
    print(f"Correcness test 1D-DFT: Numpy vs Ours custom returns {np.allclose(dft_expected, dft_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences in inverse DFT
    test_array = np.random.rand(512)
    dft_inverse_expected = np.fft.ifft(test_array)
    dft_inverse_actual = dft_inverse(test_array)
    difference = np.abs(dft_inverse_expected - dft_inverse_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 4 ----------------------")
    print(f"Correcness test 1D-FFT-Inverse: Numpy vs Ours custom returns {np.allclose(dft_inverse_expected, dft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")
    print("----------------------------------------------")

    return None

if __name__ == "__main__":
    
    mode = "1"
    image = "moonlanding.png"

    if len(sys.argv) - 1 == 2:
        mode = sys.argv[1]
        image = sys.argv[2]
    elif len(sys.argv) -1 == 1: 
        mode = sys.argv[1]
    
    if mode == "1":
        default_mode()

    elif mode == "2":
        second_mode()

    elif mode == "3":
        third_mode()

    elif mode == "4":
        fourth_mode()

    else:
        # For testing purposes
        print("Testing Correctness")
        test_correctness()



