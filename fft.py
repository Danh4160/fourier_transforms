import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time


def dft(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    c = np.exp(-1j * 2 * np.pi * k * n / N)
    return np.dot(vector, c)

def dft_inverse(vector):
    N = len(vector)
    n = np.arange(N)
    k = np.reshape(n, (N, 1))
    c = np.exp(1j * 2 * np.pi * k * n / N)
    return (1/N) * np.dot(vector, c)


def fft_inverse(vector):
    # Divide and conquer
    N = vector.shape[0]

    if (N & N - 1):
        print("Size must be a power of 2, Exiting...")
        exit()
    if N <= 16:
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
    N = vector.shape[0]
    if (N & N - 1):
        print("Size must be a power of 2, Exiting...")
        exit()
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


def trans_2d(img, test=False, mode=dft):
    imgvector = np.asarray(img)
    imgvectorshape = imgvector.shape
    M = imgvectorshape[0]
    N = imgvectorshape[1]
    
    if mode == fft:
        # Need padding
        mpower = find_next_power_2(M)
        npower = find_next_power_2(N)
        mPadd = mpower - M
        nPadd = npower - N
        N += nPadd
        M += mPadd
        imgvector = np.pad(imgvector, pad_width=((0, mPadd), (0, nPadd)))

    resultVector = np.zeros((M, N), dtype=complex)
    
    for m in range(0,M):
        resultVector[m] = mode(imgvector[m])
    for n in range(0,N):
        resultVector[:,n] = mode(resultVector[:,n])

    if mode == fft_inverse:
        if test:
            resultVector = (1/(N * M)) * resultVector[:M, :N] # Use the size of the array passed in testing
            return resultVector
        else:
            resultVector = (1/(N * M)) * resultVector[:originalM, :originalN] # Use the size of the original image
            return resultVector

    return resultVector
   

def find_next_power_2(n):
    n = n - 1
    while n & n - 1:
        n = n & n - 1       
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
            start = time.time()
            trans_2d(arr, mode=dft)
            end = time.time()
            dft_time_res.append(end - start)

            # Testing FFT
            start = time.time()
            trans_2d(arr, mode=fft)
            end = time.time()
            fft_time_res.append(end - start)

        dft_final.append(dft_time_res)
        fft_final.append(fft_time_res)

    dft_mean = np.mean(dft_final, axis=1)
    dft_std = np.std(dft_final, axis=1)
    fft_mean = np.mean(fft_final, axis=1)
    fft_std = np.std(fft_final, axis=1)

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
        fft_image_copy[int(M * t):, :int(M * (1-t))] = 0
        fft_image_copy[:,int(N * t):int(N * (1-t))] = 0

        reconstructed_image =  trans_2d(fft_image_copy, mode=fft_inverse).real
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

        reconstructed_image =  trans_2d(fft_image_copy, mode=fft_inverse).real
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
        reconstructed_image =  trans_2d(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent Low Frequency Removed in each row: {}%".format(100 * t),fontsize=22)
        plt.savefig(f'./mode_2_results/test3_{t}_with16threshold.png')
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

        reconstructed_image =  trans_2d(fft_image_copy, mode=fft_inverse).real
        plt.figure(figsize=(15,5))
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
        plt.title("Denoised"), plt.xticks([]), plt.yticks([])
        plt.suptitle("Percent Low/High Frequency Removed in each row: {}%".format(100 * t),fontsize=22)
        plt.savefig(f'./mode_2_results/test4_{t}_with16threshold.png')
    plt.close()
    return None

def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed
    print("First mode")
    fft_image = trans_2d(img, mode=fft)
    
    plt.figure(figsize=(15,5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(fft_image.real), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("Fourier Transform"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Side by side comparison",fontsize=22)
    plt.savefig("./mode_1_results/originalvsft.png")
    plt.show()
    
    return None

def second_mode():
    # Denoise an image by applying an FFT
    print("Second mode")
    
    # Use FFT
    fft_image = trans_2d(img, mode=fft)
   
    # Remove high frequencies in the middle of the plot, keeping the corners only
    optimal_threshold = 0.07
    fft_image_copy = np.copy(fft_image)
    M, N = fft_image_copy.shape
    fft_image_copy[int(M * optimal_threshold):, :int(M * (1-optimal_threshold))] = 0
    fft_image_copy[:,int(N * optimal_threshold):int(N * (1-optimal_threshold))] = 0

    number_of_nonzero = np.count_nonzero(fft_image_copy)
    print(f"Number of nonzero Fourier coefficients for denoising = {number_of_nonzero}")
    print(f"Fraction of nonzero Fourier coefficients = {number_of_nonzero / (M*N)}")

    reconstructed_image =  trans_2d(fft_image_copy, mode=fft_inverse).real
    plt.figure(figsize=(15,5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
    plt.title("Denoised"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Percent High Frequencies Removed: {}".format(100 * optimal_threshold),fontsize=22)
    plt.savefig('./mode_2_results/originalVSdenoised.png')
    plt.show()
    plt.close()

    # For testing purposes
    # second_mode_test1(fft_image)

    # second_mode_test2(fft_image)

    # second_mode_test3(fft_image)

    # second_mode_test4(fft_image)

    return None

def third_mode():
    # Compressing and saving image
    print("Third mode")
    a = 2  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure()
    plt.suptitle("Compression of Image")
    array_img = np.asarray(img)
    # print(array_img.shape)
    fft_img = trans_2d(img, mode=fft)
    fft_sorted = np.sort(np.abs(fft_img.reshape(-1)))
    for compress_perc in (1,0.8,0.6,0.4,0.2,0.05):
        thresh = fft_sorted[int(np.floor((1-compress_perc) * len(fft_sorted)))]
        ind = np.abs(fft_img)>thresh
        low_Farray = fft_img * ind
        lofArray = trans_2d(low_Farray, mode=fft_inverse).real
        #plt.figure()
        number_of_nonzero = np.count_nonzero(low_Farray)
        savearray = np.asarray(low_Farray)

        sizelowarray = low_Farray.shape
        sizex = sizelowarray[0]
        sizey = sizelowarray[1]
        np.savetxt('./mode_3_results/data'+str(c)+'.csv',savearray,delimiter=',')
        plt.subplot(a,b,c)
        plt.imshow(lofArray, plt.cm.gray)
        plt.title('Compression =' + str(100 - compress_perc * 100) + '%')
        print('Number of nonzero Fourier coefficients for Compression = '+ str(100 - compress_perc * 100) + '%' + ' is equal to: ' \
              + str(number_of_nonzero) + ' and Sparsity is equal to: ' + str(1-(number_of_nonzero/(sizex*sizey))))
        c=c+1
    plt.savefig("./mode_3_results/compression.png")
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
    fft_actual = trans_2d(img, mode=fft)
    fft_expected = np.fft.fft2(img, fft_actual.shape)
    difference = np.abs(fft_expected - fft_actual)
    s = np.sum(difference) / (img.shape[0] * img.shape[1])
    print("---------------- TEST 1 ----------------------")
    print(f"Correcness test 2D-FFT: Numpy vs Ours returns {np.allclose(fft_expected, fft_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences in inverse FFT
    test_array = np.random.rand(512, 512) 
    fft_inverse_expected = np.fft.ifft2(test_array)
    fft_inverse_actual = trans_2d(test_array, test=True, mode=fft_inverse)
    difference = np.abs(fft_inverse_expected - fft_inverse_actual)
    s = np.sum(difference) / (512 * 512)
    print("---------------- TEST 2 ----------------------")
    print(f"Correcness test 2D-FFT-Inverse: Numpy vs Ours custom returns {np.allclose(fft_inverse_expected, fft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences using DFT
    dft_actual = trans_2d(img, mode=dft)
    dft_expected = np.fft.fft2(img, s=dft_actual.shape)
    difference = np.abs(dft_expected - dft_actual)
    s = np.sum(difference) / (img.shape[0] * img.shape[1])
    print("---------------- TEST 3 ----------------------")
    print(f"Correcness test 2D-DFT: Numpy vs Ours returns {np.allclose(dft_expected, dft_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences in inverse FFT
    test_array = np.random.rand(512, 512) 
    dft_inverse_expected = np.fft.ifft2(test_array)
    dft_inverse_actual = trans_2d(test_array, mode=dft_inverse)
    difference = np.abs(dft_inverse_expected - dft_inverse_actual)
    s = np.sum(difference) / (512 * 512)
    print("---------------- TEST 4 ----------------------")
    print(f"Correcness test 2D-DFT-Inverse: Numpy vs Ours custom returns {np.allclose(dft_inverse_expected, dft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences using DFT
    test_array = np.random.rand(512)
    dft_expected = np.fft.fft(test_array)
    dft_actual = dft(test_array)
    difference = np.abs(fft_expected - fft_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 5 ----------------------")
    print(f"Correcness test 1D-DFT: Numpy vs Ours custom returns {np.allclose(dft_expected, dft_actual)}")
    print(f"Average difference between expected and actual: {s}")

    # Test for differences in inverse DFT
    test_array = np.random.rand(512)
    dft_inverse_expected = np.fft.ifft(test_array)
    dft_inverse_actual = dft_inverse(test_array)
    difference = np.abs(dft_inverse_expected - dft_inverse_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 6 ----------------------")
    print(f"Correcness test 1D-DFT-Inverse: Numpy vs Ours custom returns {np.allclose(dft_inverse_expected, dft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")
 
    # Test for differences in FFT
    test_array = np.random.rand(512)
    fft_expected = np.fft.fft(test_array)
    fft_actual = fft(test_array)
    difference = np.abs(fft_expected - fft_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 7 ----------------------")
    print(f"Correcness test 1D-FFT: Numpy vs Ours custom returns {np.allclose(fft_expected, fft_actual)}")
    print(f"Average difference between expected and actual: {s}")
   
    # Test for differences in inverse FFT
    test_array = np.random.rand(512)
    fft_inverse_expected = np.fft.ifft(test_array)
    fft_inverse_actual = fft_inverse(test_array) / 512 # Need to divide by the length of the arrray for the inverse, cannot do this in recursive algo
    difference = np.abs(fft_inverse_expected - fft_inverse_actual)
    s = np.sum(difference) / 512
    print("---------------- TEST 8 ----------------------")
    print(f"Correcness test 1D-FFT-Inverse: Numpy vs Ours custom returns {np.allclose(fft_inverse_expected, fft_inverse_actual)}")
    print(f"Average difference between expected and actual: {s}")

    return None


def test_reconstruct_image():
    # Apply 2D FFT then apply 2D FFT Inverse
    img_array = np.asarray(img)
    fft_image = trans_2d(img_array, mode=fft)
    reconstructed_image = trans_2d(fft_image, mode=fft_inverse)
    difference = np.abs(img_array - reconstructed_image)
    s = np.sum(difference) / (originalN * originalM)
    print("---------------- TEST 9 ----------------------")
    print(f"Original Image and Reconstructed image are similar using 2D FFT: {np.allclose(img_array, reconstructed_image)}")
    print(f"Average difference between orginal image and reconstructed image: {s}")

    # Apply 2D FFT then apply 2D DFT Inverse
    img_array = np.asarray(img)
    dft_image = trans_2d(img_array, mode=dft)
    reconstructed_image = trans_2d(dft_image, mode=dft_inverse)
    difference = np.abs(img_array - reconstructed_image)
    s = np.sum(difference) / (originalN * originalM)
    print("---------------- TEST 10 ----------------------")
    print(f"Original Image and Reconstructed image are similar using 2D DFT: {np.allclose(img_array, reconstructed_image)}")
    print(f"Average difference between orginal image and reconstructed image: {s}")

    # Apply 1D DFT then apply 1D DFT Inverse
    test_array = np.random.rand(512)
    dft_image = dft(test_array)
    reconstructed_image = dft_inverse(dft_image)
    difference = np.abs(test_array - reconstructed_image)
    s = np.sum(difference) / 512
    print("---------------- TEST 11 ----------------------")
    print(f"Original Image and Reconstructed image are similar using 1D DFT: {np.allclose(test_array, reconstructed_image)}")
    print(f"Average difference between orginal image and reconstructed image: {s}")
   
    # Apply 1D FFT then apply 1D FFT Inverse
    test_array = np.random.rand(512)
    dft_image = fft(test_array)
    reconstructed_image = fft_inverse(dft_image) / 512 # Need this division for FFT inverse
    difference = np.abs(test_array - reconstructed_image)
    s = np.sum(difference) / 512
    print("---------------- TEST 12 ----------------------")
    print(f"Original Image and Reconstructed image are similar using 1D FFT: {np.allclose(test_array, reconstructed_image)}")
    print(f"Average difference between orginal image and reconstructed image: {s}")
    print("----------------------------------------------")

def test_numpyFT_and_personalFT():
    print("---------------- TEST 13 ----------------------")
    print(f" Difference between Numpy FFT and our FFT")
    print("----------------------------------------------")
    fft_image = trans_2d(img, mode=fft)
    fft_image2 = np.fft.fft2(img, fft_image.shape) 
    plt.figure(figsize=(15,5))
    plt.subplot(121), plt.imshow(np.abs(fft_image.real), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("Our Fourier Transform"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(fft_image2.real), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("Numpy  Fourier Transform"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Side by side comparison",fontsize=22)
    plt.savefig("./mode_1_results/ComparingPlots.png")
    plt.show()
 



if __name__ == "__main__":
    # Default values
    mode = "1"
    image = "moonlanding.png"


    if len(sys.argv) > 6:
        print("Too many arguments, Exiting...")
        exit()

    if "-m" in sys.argv and "-i" in sys.argv and len(sys.argv) != 6:
        print("Missing arguments, Exiting...")
        exit()

    if '-m' not in sys.argv and '-i' not in sys.argv and len(sys.argv) != 2:
        print("Bad Command, Exiting...")
        exit()

    
    for i in range(len(sys.argv)):
        if  sys.argv[i] == '-m':
            i += 1
            mode = sys.argv[i]
            
        elif sys.argv[i] == '-i':
            i += 1
            image = sys.argv[i]
    try:
        img = plt.imread(f'./{image}').astype(float)
        originalM, originalN = np.asarray(img).shape
    except Exception:
        print("Unable to open image, Exiting...")
        exit()
    
    if mode == "1":
        default_mode()

    elif mode == "2":
        second_mode()

    elif mode == "3":
        third_mode()

    elif mode == "4":
        fourth_mode()

    elif mode == "t":
        # Testing purposes
        print("Testing...")
        test_correctness()
        test_reconstruct_image()
        test_numpyFT_and_personalFT()
    else:
        print("Bad Argument, Exiting...")



