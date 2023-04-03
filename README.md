# Command Line tool For Fourier Transforms
Written by Dan Hosi and Harsh Patel

## How to use the tool
In the command line, `python fft.py [-m mode] [-i image]` where the argument isdefined as follows: 

• mode (optional): 
  - [1] (Default) for fast mode where the image is converted into its FFT form and displayed
  - [2] for denoising where the image is denoised by applying an FFT, truncating high 
frequencies and then displayed
  - [3] for compressing and saving the image
  - [4] for plotting the runtime graphs for the report 
  - [t] for us developers to run certain tests for our algorithms.
  
• image (optional): filename of the image we wish to take the DFT of. (Default: moonlanding.png)
https://github.com/Danh4160/fourier_transforms/blob/main/moonlanding.png


