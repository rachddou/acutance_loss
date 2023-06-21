
import os
import argparse
from time import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.models import FFDNet
from src.acutance_loss import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from src.utils import batch_ssim, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb, compute_noise_map,rgb_2_grey_tensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('PerceptualImageError/model/')
sys.path.append('PerceptualImageError/utils/')
from PerceptualImageError.utils.image_utils import *
from PerceptualImageError.model.PieAPPv0pt1_PT import PieAPP
import skimage.io as skio

def test_ffdnet_dataset(**args):
    torch.manual_seed(0)
    r"""Denoises an input image with FFDNet
    """
    # Init logger
    logger = init_logger_ipol()
    # Check if input exists and if it is RGB
    im_files = [os.path.join(args['input'], f) for f in os.listdir(args['input']) if os.path.isfile(os.path.join(args['input'], f))]
    try:
        rgb_den = is_rgb(im_files[0])
    except:
        raise Exception('Could not open the input image')
    N_multi = args['Nmulti']

    if rgb_den:
        in_ch = 3
        model_fn = args['model_path']   
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = args['model_path']
    #initialize global metrics
    psnr_test_set = 0
    ssim_test_set = 0
    acutance_test_set = 0
    PieAPP_test_set = 0
    PieAPP_test_set_n = 0
    acutance_test_set = 0
    noise_test = True
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                model_fn)

    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights
    if args['cuda']:
        if 'ckpt' in model_fn:
            state_dict = torch.load(model_fn)['state_dict']
        else:
            state_dict = torch.load(model_fn)
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


    masks = compute_masks(100,5)
        ## spectrum_theory computation
        # theory_spectrum = test_spec(theoretical_spectrum(200,compute = False),masks)
    csf_vals = csf(50,5).unsqueeze(0)
        #masks = masks.cuda()
    for filename in im_files:
        # Open image as a CxHxW torch.Tensor

        imorig = cv2.imread(filename)
        # from HxWxC to CxHxW, RGB image
        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)

        imorig = np.expand_dims(imorig, 0)
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
        imorig = normalize(imorig)
        if args['contrast_reduction']:
                for i in range(3):
                    imorig[:,i,...] = 0.18*(imorig[:,i,...] -imorig[:,i,...].min())/(1e-8+np.abs(imorig[:,i,...].max() -imorig[:,i,...].min()))
                    imorig[:,i,...] = imorig[:,i,...]+0.18-imorig[:,i,...].mean()    
                imorig = np.clip(imorig,0,1)
        imorig = torch.Tensor(imorig)
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
            nsigma = Variable(torch.FloatTensor([args['noise_sigma']]).type(dtype))

        # Measure runtime
        start_t = time()

        # Estimate noise and subtract it to the input image



        noise_map = compute_noise_map(imnoisy, nsigma, mode = args["noise_map"])
            #noise_map = compute_noise_map(imnoisy, nsigma, mode = args['noise_map'])


        im_noise_estim = model(imnoisy, noise_map)
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
        
        fname = filename.split("/")
        fname = fname[-1].split(".")[0]
        print(fname)
        outimg = np.float64(variable_to_cv2_image(outim))
        noiseimg = variable_to_cv2_image(imnoisy)
        img_orig = variable_to_cv2_image(imorig)
        residual = (img_orig - outimg)
        residual  = (residual -residual.min())/(residual.max() -residual.min())*255

        outimg = np.uint8(np.clip(outimg,0,255))
        psnr = compare_psnr(outimg, img_orig)
        psnr_test_set+= psnr
            
        if args['save']:
            if not(os.path.isdir(os.path.join("tests",args["save_dir"]))):
                os.makedirs(os.path.join("tests",args["save_dir"]))
            cv2.imwrite("tests/"+args["save_dir"]+fname+"_ffdnet.png", outimg)
            if args['save_res']:
                cv2.imwrite("tests/"+args["save_dir"]+fname+"_residual.png", residual)
            #cv2.imwrite("tests/"+args["save_dir"]+fname+"_noisy.png", noiseimg)
        # Compute PSNR and log it
        if rgb_den:
            logger.info("### RGB denoising ###")
        else:
            logger.info("### Grayscale denoising ###")
        if args['add_noise']:
            ssim_val = batch_ssim(outim,imorig)
            ssim_test_set+=ssim_val
            if args['acutance']:
                outim_grey  = rgb_2_grey_tensor(outim.data.cpu(), False)
                imorig_grey  = rgb_2_grey_tensor(imorig.data.cpu(), False)

                acutance_img = image_acutance(outim_grey,imorig_grey,masks,csf_vals).mean()
                print(acutance_img)
                acutance_test_set +=acutance_img
            if args['pieapp']:
                ######## variables
                patch_size = 64
                batch_size = 1
                stride_val = 6
                num_patches_per_dim = 10

                _,ch,rows,cols = imorig.size()
                #print(imorig.shape)
                #print(cols)
                y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val),np.array([rows - patch_size])), axis=0)
                num_y = len(y_loc)
                x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val),np.array([cols - patch_size])), axis=0)
                num_x = len(x_loc)

                ######## initialize the model
                PieAPP_net = PieAPP(int(batch_size),int(num_patches_per_dim))
                PieAPP_net.load_state_dict(torch.load('PerceptualImageError/weights/PieAPPv0.1.pth'))

                PieAPP_net.cuda()

                score_accum = 0.0
                weight_accum = 0.0
                score_accum_2 = 0.0
                weight_accum_2 = 0.0
                #iterate through smaller size sub-images (to prevent memory overload)
                for x_iter in range(0, -(-num_x//num_patches_per_dim)):
                    for y_iter in range(0, -(-num_y//num_patches_per_dim)):
                        # compute the size of the subimage
                        if (num_patches_per_dim*(x_iter + 1) >= num_x):
                            size_slice_cols = cols - x_loc[num_patches_per_dim*x_iter]
                        else:
                            size_slice_cols = x_loc[num_patches_per_dim*(x_iter + 1)] - x_loc[num_patches_per_dim*x_iter] + patch_size-stride_val
                        if (num_patches_per_dim*(y_iter + 1) >= num_y):
                            size_slice_rows = rows - y_loc[num_patches_per_dim*y_iter]
                        else:
                            size_slice_rows = y_loc[num_patches_per_dim*(y_iter + 1)] - y_loc[num_patches_per_dim*y_iter] + patch_size - stride_val
                        # obtain the subimage and samples patches
                        A_sub_im = outim[:,:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols]
                        B_sub_im = imnoisy[:,:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols]
                        ref_sub_im = imorig[:,:, y_loc[num_patches_per_dim*y_iter]:y_loc[num_patches_per_dim*y_iter]+size_slice_rows, x_loc[num_patches_per_dim*x_iter]:x_loc[num_patches_per_dim*x_iter]+size_slice_cols]


                        A_tensor = A_sub_im.unfold(2,64,stride_val).unfold(3,64,stride_val)
                        A_tensor = A_tensor.contiguous().view( 3,-1,64,64).permute((1,0,2,3))
                        B_tensor = B_sub_im.unfold(2,64,stride_val).unfold(3,64,stride_val)
                        B_tensor = B_tensor.contiguous().view( 3,-1,64,64).permute((1,0,2,3))
                        ref_tensor = ref_sub_im.unfold(2,64,stride_val).unfold(3,64,stride_val)
                        ref_tensor = ref_tensor.contiguous().view(3,-1,64,64).permute((1,0,2,3)) 

                        # A_tensor = torch.nn.Unfold(kernel_size=(64,64), stride = stride_val)(A_sub_im)
                        # ref_tensor = torch.nn.Unfold(kernel_size=(64,64), stride = stride_val)(ref_sub_im)
                        N_patches = A_tensor.size()[0]
                        A_patches_var = 255.*A_tensor
                        B_patches_var = 255.*B_tensor
                        ref_patches_var = 255.*ref_tensor
                        A_patches_var = torch.round(A_patches_var)
                        B_patches_var = torch.round(B_patches_var)
                        ref_patches_var = torch.round(ref_patches_var)
                        A_patches_var = A_patches_var.cuda()
                        B_patches_var = B_patches_var.cuda()
                        ref_patches_var = ref_patches_var.cuda()
                        num_patches_curr = int(N_patches/batch_size)
                        PieAPP_net.num_patches = num_patches_curr
                        # forward pass 
                        score_1, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(A_patches_var.float(), ref_patches_var.float())
                        curr_err = PieAPP_patchwise_errors.cpu().data.numpy()
                        curr_weights = 	PieAPP_patchwise_weights.cpu().data.numpy()
                        score = np.multiply(curr_err,curr_weights)
                        score_accum += np.sum(score)
                        weight_accum += np.sum(curr_weights)
                        if noise_test :
                            score_1, PieAPP_patchwise_errors, PieAPP_patchwise_weights = PieAPP_net.compute_score(B_patches_var.float(), ref_patches_var.float())
                            curr_err = PieAPP_patchwise_errors.cpu().data.numpy()
                            curr_weights = 	PieAPP_patchwise_weights.cpu().data.numpy()
                            score = np.multiply(curr_err,curr_weights)
                            score_accum_2 += np.sum(score)
                            weight_accum_2 += np.sum(curr_weights)

                PieAPP_val = score_accum/weight_accum
                PieAPP_test_set+=PieAPP_val
            #     if noise_test :
            #         PieAPP_val = score_accum_2/weight_accum_2
            #         PieAPP_test_set_n+=PieAPP_val
            #psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
            #nlp_val = batch_nlp(outim,imorig)
            #print(ssim_val)
            #print(nlp_val)
            #print(psnr)
            #print((255**2)*10**(-(psnr/10)))

            #logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
            # logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
        else:
            logger.info("\tNo noise was added, cannot compute PSNR")
        logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))
    psnr_test_set*= (1./len(im_files))
    ssim_test_set*= (1./len(im_files))
    if args['pieapp']:
        PieAPP_test_set*= (1./len(im_files))
    print("SSIM : {}".format(ssim_test_set))
    print("PSNR : {}".format(psnr_test_set))
    if args['acutance']:
        acutance_test_set*= (1./len(im_files))
        print("ACUTANCE : {}".format(acutance_test_set))
    if args['pieapp']:
        print("PieAPP : {}".format(PieAPP_test_set))
        if noise_test:
            PieAPP_test_set_n*= (1./len(im_files))
            print("PieAPP noisy : {}".format(PieAPP_test_set_n))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="True")
    parser.add_argument("--input", type=str, default="datasets/test_sets/Kodak24/", \
                        help='path to input dataset')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, \
                        help='noise level used on test set')
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    parser.add_argument("--model_path","--p", type = str, default = "TRAINING_LOGS/mix_acutance/ckpt.pth")
    parser.add_argument("--multi", action='store_true', \
                        help="different realisations of the noise")
    parser.add_argument("--Nmulti", type = int, default = 7, \
                        help=" number of different realisations of the noise")
    parser.add_argument("--noise_map", type = str, default = "constant", \
                        help=" noise map mode")
    parser.add_argument("--save", action='store_true', \
                        help="save denoised and noisy images")
    parser.add_argument("--save_res", action='store_true', \
                        help="save residual images")
    parser.add_argument("--save_dir", type = str, default = "dl_denoising/", \
                        help=" noise map mode")                        
    parser.add_argument("--pieapp", action='store_true', \
                        help="true if we want to measure the pieapp metric")
    parser.add_argument("--acutance", action='store_true', \
                        help="true if we want to measure the acutance")
    parser.add_argument("--grey_color", action='store_true', \
                        help="true if we want to denoise each canal with the grey level trained model")
    parser.add_argument("--contrast_reduction", action='store_true', \
                        help="true if you want to reduce contrast of input images")
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    print(torch.cuda.is_available())
    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_ffdnet_dataset(**vars(argspar))
