import numpy as np 
import torch
import skimage.io as skio
import matplotlib.pyplot as plt

def compute_masks(w,interval = 1):
    """
    computes circular masks for an image of width w with an interval=interval 
    """
    lim = np.uint8(w/(2*interval))
    res = torch.zeros((lim,w,w)).int()
    X = torch.arange(0,100)-49.5
    X,Y = torch.meshgrid(X,X)
    for i in range(lim):
        res[i,...] = (X**2 + Y**2 < (interval*(i+1))**2).int()
        res[i,...] -= (X**2 + Y**2 < (interval*i)**2).int()
    return(res)


def csf(n,interval,mode = "original"):
    """
    contrast sensitivity function as explained in the paper of Cao et al.
    mode == "original" --> Cao's sensitivity function
    mode == "constant" --> constant sensitivity function =1  
    """
    if mode == "original":
        b = 0.2
        c = 0.8
        x = torch.arange(interval/2,n,interval)
        x = (40*torch.abs(x))/n
        y = (x**c)*torch.exp(-b*x)
    elif mode == "constant":
        x = torch.arange(interval/2,n,interval)
        s = x.shape[0]
        y = torch.ones(s)
    a = torch.sum(y)
    y = y/a
    return(y)

def acutance(output,input,masks,csf_vals,single_channel = False,train = True, new_mtf = True):
    """
    acutance function that combines the CSF and the MTF computation
    output : torch Tensor
    input : torch Tensor
    masks : torch Tensor version of the computed masks
    csf_vals : torch Tensor of the values of the CSF
    single_channel : whether to compute the acutance channel per channel or on a single grey level channel
    train : uses cuda acceleration
    new_mtf : uses the MTF computation presented by Artmann in 2015
    """
    # declare important dimension variables
    w = masks.shape[0]
    n = output.shape[0]
    if single_channel:
        n_dims = 1
    else:
        n_dims = 3

    # declare cuda or cpu variables depending on the used device
    if train :
        spectrum_out = torch.zeros_like(output,dtype =torch.cfloat).cuda()
        spectrum_in = torch.zeros_like(input,dtype =torch.cfloat).cuda()
        ratio_1d = torch.zeros((n,n_dims,w)).cuda()
    else : 
        spectrum_out = torch.zeros_like(output,dtype =torch.cfloat)
        spectrum_in = torch.zeros_like(input,dtype =torch.cfloat)
        ratio_1d = torch.zeros((n,n_dims,w))

    # compute the spectrum of the input image and the ground truth image
    for i in range(n_dims):
        spectrum_out[:,i,:,:] = torch.fft.fftshift(torch.fft.fft2(output[:,i,:,:]),dim = (1,2))
        spectrum_in[:,i,:,:] = torch.fft.fftshift(torch.fft.fft2(input[:,i,:,:]),dim = (1,2))
    
    # compute the 2D MTF with the new formula
    if new_mtf:
        ratio_2d = torch.abs(torch.real(spectrum_out*spectrum_in.conj()) / (1e-5+spectrum_in*spectrum_in.conj()))
    else:
        ratio_2d = torch.abs(spectrum_out)**2/(1e-5+torch.abs(spectrum_in)**2)
    
    # compute the 1D MTF with ring mean
    for k in range(0,w):
        for l in range(n_dims):
            val_ratio_1d = torch.sum(ratio_2d[:,l,:,:]*masks[k].unsqueeze(0), dim = (1,2))
            weight = torch.sum(masks[k])
            if weight  ==0:
                weight = 1e-8
            val_ratio_1d = val_ratio_1d/weight
            ratio_1d[:, l,k] = val_ratio_1d
    # compute the acutance score as the integral of the dot product of the MTF and the CSF

    acutance_val = torch.abs(1-torch.sum(ratio_1d[:,:,:]*csf_vals[:], dim = 2))
    return(acutance_val)

def acutance_prev_mtf(output,input,masks,csf_vals,single_channel = False,train = True):
    w = masks.shape[0]
    n = output.shape[0]
    if single_channel:
        n_dims = 1
    else:
        n_dims = 3
    if train :
        spectrum_out = torch.zeros_like(output).cuda()
        spectrum_in = torch.zeros_like(input).cuda()
    else : 
        spectrum_out = torch.zeros_like(output)
        spectrum_in = torch.zeros_like(input)

    for i in range(n_dims):
        spectrum_out[:,i,:,:] = torch.abs(torch.fft.fftshift(torch.fft.fft2(output[:,i,:,:]),dim = (1,2)))**2
        spectrum_in[:,i,:,:] = torch.abs(torch.fft.fftshift(torch.fft.fft2(input[:,i,:,:]),dim = (1,2)))**2

    if train:
        spec_1d_in = torch.zeros((n,n_dims,w)).cuda()
        spec_1d_out = torch.zeros((n,n_dims,w)).cuda()
    else:
        spec_1d_in = torch.zeros((n,n_dims,w))
        spec_1d_out = torch.zeros((n,n_dims,w))
    for k in range(0,np.uint8(1*w)):
        for l in range(n_dims):
            val_in = torch.sum(spectrum_in[:,l,:,:]*masks[k].unsqueeze(0), dim = (1,2))
            val_out = torch.sum(spectrum_out[:,l,:,:]*masks[k].unsqueeze(0), dim = (1,2))
            weight = torch.sum(masks[k])
            if weight  ==0:
                weight = 1e-8
            val_in = val_in/weight
            val_out = val_out/weight
            spec_1d_in[:, l,k] = val_in
            spec_1d_out[:, l,k] = val_out

    ratio = spec_1d_out/(spec_1d_in+1e-5)
    acutance_val = torch.abs(1-torch.sum(ratio[:,:,:]*csf_vals[:], dim = 2))
    #return(acutance_val,spec_1d_in,spec_1d_out)
    return(acutance_val)

def image_acutance(output,input,masks,csf_vals):
    """
    function to compute the acutance for an image of arbitrary size
    """
    stride = 100
    w = 100
    acutance_final = 0
    n = 0
    for i in range(output.shape[2]//stride-1):
        for j in range(output.shape[3]//stride-1):
            n+=1
            ac = acutance(output[:,:, stride*i:stride*i+w,stride*j:stride*j+w],input[:,:, stride*i:stride*i+w,stride*j:stride*j+w], masks,csf_vals,single_channel = True, train = False,new_mtf=True)
            
            acutance_final+= ac
    print(acutance_final)
    return(acutance_final/n)

def theory_spectrum(w):
    """
    function that computes the theoretical spectrum of a dead leaves image as presented in Cao's paper,
    not used in these experiments
    """
    line = np.arange(-((w-1)/2),(w+1)/2)**2
    spectrum = np.zeros((w,w))
    spectrum[w//2,w//2] = 1e-9
    for i in range(w):
        spectrum[:,i]+=line
        spectrum[i,:]+=line
    spectrum = spectrum**(-1.93/2)
    return spectrum