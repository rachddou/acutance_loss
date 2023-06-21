
import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.models import FFDNet
from src.utils import batch_psnr,batch_ssim, normalize, init_logger_ipol, \
                variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb, compute_noise_map

from time import time

def test_ffdnet(**args):
    r"""Denoises an input image with FFDNet
    """
    torch.manual_seed(0)
    # Init logger
    logger = init_logger_ipol()
    # Check if input exists and if it is RGB
    try:
        rgb_den = is_rgb(args['input'])
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    if rgb_den:
        in_ch = 3
        model_fn = args['model_path']
        imorig = cv2.imread(args['input'])
        # from HxWxC to CxHxW, RGB image
        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)

    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = args['model_path']
        imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)
    print(imorig.shape)
    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2]%2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, \
                imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3]%2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, \
                imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
    print(imorig.shape)
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                model_fn)

    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)
    # Load saved weights
    if args['cuda']:
        state_dict = torch.load(model_fn)['state_dict']

        model = nn.DataParallel(net).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Add noise
    if args['add_noise']:

        noise = torch.FloatTensor(imorig.size()).\
            normal_(mean=0, std=args['noise_sigma'])
        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()
        # Test mode
    with torch.no_grad(): # PyTorch v0.4.0

        imorig, imnoisy = Variable(imorig.type(dtype)), \
            Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([args['ratio']*args['noise_sigma']]).type(dtype))

    # Measure runtime
    start_t = time()

    # Estimate noise and subtract it to the input image

    if args["noise_map"] == "texture":
        n_map_1 = compute_noise_map(imnoisy, 0.8*nsigma, mode = "constant") 
        oracle = model(imnoisy, n_map_1)
        oracle = torch.clamp(imnoisy-oracle, 0., 1.)
        noise_map = compute_noise_map(oracle, nsigma, mode = "texture")
        #noise_map = compute_noise_map(imorig, nsigma, mode = "texture")
    else:
        noise_map = compute_noise_map(imnoisy, nsigma, mode = args["noise_map"])
    im_noise_estim = model(imnoisy, noise_map)
    if args["not_residual"]:
        outim = torch.clamp(im_noise_estim, 0., 1.)
    else:
        outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
    stop_t = time()

    if expanded_h:
        imorig = imorig[:, :, :-1, :]
        outim = outim[:, :, :-1, :]
        #imnoisy = imnoisy[:, :, :-1, :]

    if expanded_w:
        imorig = imorig[:, :, :, :-1]
        outim = outim[:, :, :, :-1]
        #imnoisy = imnoisy[:, :, :, :-1]

    # Compute PSNR and log it
    if rgb_den:
        logger.info("### RGB denoising ###")
    else:
        logger.info("### Grayscale denoising ###")
    if args['add_noise']:
        psnr = batch_psnr(outim, imorig, 1.)
        ssim_val = batch_ssim(outim,imorig)
        #psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
        #nlp_val = batch_nlp(outim,imorig)
        print(ssim_val)
        #print(nlp_val)
        print(psnr)
        print((255**2)*10**(-(psnr/10)))

        #logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
        logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
    else:
        logger.info("\tNo noise was added, cannot compute PSNR")
    # Compute difference
    diffout   = 2*(outim - imorig) + .5
    if args["not_residual"]:
            diffnoise = 2*(imnoisy-imorig) + .5

    # Save images
    if not args['dont_save_results']:
        #noisyimg = variable_to_cv2_image(imnoisy)
        outimg = variable_to_cv2_image(outim)
        filename = args['filename']
        #out_nm = variable_to_cv2_image(noise_map)
        if args["noise_map"] == "texture":
            out_nm = noise_map.cpu().detach().numpy()[0].transpose(1,2,0)
            out_nm = 255*(out_nm - out_nm.min())/(out_nm.max()-out_nm.min())
            cv2.imwrite("tests/"+filename+"_noise_map.png", out_nm)		
        #cv2.imwrite("tests/noisy.png", noisyimg)
        cv2.imwrite("tests/"+filename+"_ffdnet.png", outimg)
        if args['save_noisy']:
            out_noisy = variable_to_cv2_image(torch.clamp(imnoisy,0,1))
            cv2.imwrite("tests/"+filename+"_noisy.png", out_noisy)
        if args['add_noise']:
            #cv2.imwrite("tests/"+filename+"_noisy_diff_{}.png", variable_to_cv2_image(diffnoise))
            #cv2.imwrite("tests/"+filename+"_ffdnet_diff.png", variable_to_cv2_image(diffout))
            pass

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="True")
    parser.add_argument("--input", type=str, default="input.png", \
                        help='path to input image')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, \
                        help='noise level used on test set')
    parser.add_argument("--ratio", type=float, default=1., \
                        help='noise level used on test set')
    parser.add_argument("--dont_save_results","--d", action='store_true', \
                        help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    parser.add_argument("--filename","--f", type=str, default = "mse", \
                        help="filename of the saved image")
    parser.add_argument("--model_path","--p", type = str, default = "TRAINING_LOGS/dl_rmin_1_logs/ckpt.pth")

                        
    parser.add_argument("--save_noisy", action='store_true', \
                        help="saves the noisy image as well")
    parser.add_argument("--not_residual","--r", action = "store_true",\
                                                help = "stores if the model was trained residually or not")
    parser.add_argument("--noise_map", type = str, default = "constant", \
                        help=" noise map mode")
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')



    test_ffdnet(**vars(argspar))
