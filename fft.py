import sys
import numpy
import matplotlib as plt


def default_mode():
    # Fast Mode where image is converted to its FFT form and displayed

    return None


def second_mode():
    # Denoise an image by applying an FFT
    return None

def third_mode():
    # Compressing and saving image
    return None

def fourth_mode():
    # Plotting runtime graphs for report
    return None


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}:{arg}")
    arg_list = sys.argv
    

