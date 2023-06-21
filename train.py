
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils

from tensorboardX import SummaryWriter
from src.acutance_loss import *
from src.models import FFDNet
from src.dataset import HDF5Dataset
from src.utils import weights_init_kaiming, batch_psnr, init_logger, \
			svd_orthogonalization, compute_noise_map, rgb_2_grey_tensor


from torch.utils.data.dataset import ConcatDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    torch.manual_seed(1)
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')
    dataset_train_1 = HDF5Dataset(args.filenames[0], recursive=False, load_data=True, mask = 0,data_cache_size=4, transform=None)
    if not args.natural_only:
        dataset_train_2 = HDF5Dataset(args.filenames[1], recursive=False, load_data=True, mask = 1, \
                                        contrast_reduction = args.contrast_reduction,data_cache_size=4, transform=None)
        dataset_train = ConcatDataset([dataset_train_1, dataset_train_2])
    else:
        dataset_train = dataset_train_1
    dataset_val = HDF5Dataset(args.filenames[2], recursive=False, load_data=True, data_cache_size=4, transform=None)

    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists("TRAINING_LOGS/"+args.log_dir):
        os.makedirs("TRAINING_LOGS/"+args.log_dir)
    writer = SummaryWriter("TRAINING_LOGS/"+args.log_dir)
    logger = init_logger(args)

    # Create model
    if not args.gray:
        in_ch = 3
    else:
        in_ch = 1
    
    net = FFDNet(num_input_channels=in_ch)
    # Initialize model with He init

    net.apply(weights_init_kaiming)
    # Define loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume training or start anew
    if args.resume_training:
        resumef = os.path.join("TRAINING_LOGS/"+args.log_dir, args.check_point)
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = args.epochs
            new_milestone = args.milestone
            save_every_epochs = args.save_every_epochs
            new_log_dir = args.log_dir
            current_lr = args.lr
            new_natural_only = args.natural_only
            new_no_orthog = args.no_orthog
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            args.epochs = new_epoch

            args.milestone = new_milestone
            args.lr = current_lr
            args.log_dir = new_log_dir
            args.save_every_epochs = save_every_epochs
            args.natural_only = new_natural_only
            args.no_orthog = new_no_orthog

            print("=> loaded checkpoint '{}' (epoch {})"\
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))

    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = args.no_orthog
    # Training

    interval = 5
    masks = compute_masks(100, 5)
    n_max = np.ceil(100/2*np.sqrt(2))
    n_max = 100//2
    mode = args.mode_csf
    csf_vals = csf(n_max,5,mode=mode).unsqueeze(0).cuda()
    masks = masks.cuda()
    errors_acutance = 0
    for epoch in range(start_epoch, args.epochs):
        # Learning rate value scheduling according to args.milestone
        if len(args.milestone) == 3 and epoch > args.milestone[2]:
            current_lr = args.lr / 1000.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[1]:
            current_lr = args.lr / 100.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[0]:
            current_lr = args.lr / 10.
        else:
            current_lr = args.lr

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # create a data_loader here for curriculum learning
        loader_train = DataLoader(dataset=dataset_train, num_workers=16, \
            batch_size=args.batch_size, shuffle=True)
        limit = len(dataset_train)//(args.batch_size)
        # torch.set_num_threads(1)

        print("number of batches for this epoch : {}".format(limit))
        for i, data in enumerate(loader_train, 0):
            if i>= limit :
                break
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image
            loss_mask = data[0]
            img_train = data[1]
            noise = torch.zeros(img_train.size())
            stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                            size=noise.size()[0])
            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                noise[nx, :, :, :] = torch.FloatTensor(sizen).\
                                    normal_(mean=0, std=stdn[nx])
            imgn_train = img_train + noise
            # Create input Variables
            # loss_mask = Variable(torch.mean(loss_mask.float()).cuda())
            loss_mask = Variable(loss_mask.float()).cuda()
            img_train = Variable(img_train.cuda())
            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))
            noise_map = compute_noise_map(imgn_train, stdn_var, mode = "constant")
            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            if not(args.natural_only):
                if  args.warmup:
                    if epoch > 45:
                        acutance_batch = acutance(imgn_train-out_train,img_train,masks,csf_vals,new_mtf=args.new_mtf)
                        acutance_batch = loss_mask.reshape((loss_mask.shape[0],1))*acutance_batch
                        acutance_loss = torch.mean(acutance_batch, dim = 1)
                        acutance_loss = args.lambda_acutance*torch.sum(acutance_loss)/(torch.sum(loss_mask)+1e-5)
                        loss+= acutance_loss
                else:
                    if args.single_dim_acutance:
                        grey_denoised = rgb_2_grey_tensor(imgn_train-out_train, True)
                        grey_input = rgb_2_grey_tensor(img_train, True)
                        acutance_batch = acutance(grey_denoised,grey_input,masks,csf_vals,single_channel=True,new_mtf=args.new_mtf)
                    else:
                        acutance_batch = acutance(imgn_train-out_train,img_train,masks,csf_vals,single_channel=False,new_mtf=args.new_mtf)

                    acutance_batch = loss_mask.reshape((loss_mask.shape[0],1))*acutance_batch
                    acutance_loss = torch.mean(acutance_batch, dim = 1)
                    acutance_loss = args.lambda_acutance*torch.sum(acutance_loss)/(torch.sum(loss_mask)+1e-5)
                    loss+= acutance_loss
                    indices = torch.where(acutance_batch > 1000)[0]

            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            # Results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train, noise_map), 0., 1.)
            psnr_train = batch_psnr(out_train, img_train, 1.)
            # PyTorch v0.4.0: loss.data[0] --> loss.item()

            if training_params['step'] % args.save_every == 0:

                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log the scalar values
                
                writer.add_scalar('loss', loss.data.item(), training_params['step'])
                if args.warmup: 
                    if epoch > 45:
                        writer.add_scalar('acutance_loss', acutance_loss.data.item(), training_params['step'])
                        print("[epoch %d][%d/%d] loss: %.4f acutance_loss : %.4f PSNR_train: %.4f" %\
                            (epoch+1, i+1, limit, loss.data.item(),acutance_loss.data.item(),psnr_train))
                    else:
                        print("[epoch %d][%d/%d] loss: %.4f  PSNR_train: %.4f" %\
                            (epoch+1, i+1, limit, loss.data.item(),psnr_train))

                else:
                    if not(args.natural_only):
                        writer.add_scalar('acutance_loss', acutance_loss.data.item(), training_params['step'])
                        print("[epoch %d][%d/%d] loss: %.4f acutance_loss : %.4f PSNR_train: %.4f" %\
                            (epoch+1, i+1, limit, loss.data.item(),acutance_loss.data.item(),psnr_train))
                    else:
                        print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %\
                            (epoch+1, i+1, limit, loss.data.item(),psnr_train))
                writer.add_scalar('PSNR on training data', psnr_train, \
                    training_params['step'])


            training_params['step'] += 1
        # The end of each epoch
        model.eval()

        # Validation
        psnr_val = 0
        for _,valimg in dataset_val:
            img_val = torch.unsqueeze(valimg, 0).float()
            print(img_val.size())
            noise = torch.FloatTensor(img_val.size()).\
                    normal_(mean=0, std=args.val_noiseL).float()
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            sigma_noise = Variable(torch.cuda.FloatTensor([args.val_noiseL]))
            noise_map = compute_noise_map(img_val, sigma_noise, mode = "constant")
            out_val = torch.clamp(imgn_val-model(imgn_val, noise_map), 0., 1.)
            psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', current_lr, epoch)

        # Log val images
        try:
            if epoch == 0:
                # Log graph of the model
                writer.add_graph(model, (imgn_val, sigma_noise), )
                # Log validation images
                for idx in range(2):
                    imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
                                            nrow=2, normalize=False, scale_each=False)
                    imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
                                            nrow=2, normalize=False, scale_each=False)
                    writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
                    writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
            for idx in range(2):
                imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
                                        nrow=2, normalize=False, scale_each=False)
                writer.add_image('Reconstructed validation image {}'.format(idx), \
                                imrecons, epoch)
            # Log training images
            imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
                         scale_each=True)
            writer.add_image('Training patches', imclean, epoch)

        except Exception as e:
            logger.error("Couldn't log results: {}".format(e))

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        torch.save(model.state_dict(), os.path.join("TRAINING_LOGS/"+args.log_dir, 'net.pth'))
        save_dict = { \
            'state_dict': model.state_dict(), \
            'optimizer' : optimizer.state_dict(), \
            'training_params': training_params, \
            'args': args\
            }
        filename = 'ckpt'
        
        if epoch+1 == args.epochs :
            final_filename = filename+"_back_up"
            torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, final_filename+'.pth'))
        torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, filename+'.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, \
                                      filename+'_e{}.pth'.format(epoch+1)))
        del save_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true',\
                        help='train grayscale image denoising instead of RGB')


    #Training parameters
    parser.add_argument("--batch_size","--bs",type=int, default=256, 	\
                     help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, \
                     help="Number of total training epochs")
    parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
                     help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, \
                        help='noise level used on validation set')
    parser.add_argument("--milestone",nargs = '+', type=int, default=[50, 60,80], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, \
                     help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',\
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--resume_training", action='store_true',\
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--warmup", action='store_true',\
                        help="warmup before inputing acutance loss")
    parser.add_argument("--grad_clip", action='store_true',\
                        help="perform gradient clipping")
    parser.add_argument("--single_dim_acutance", action='store_true',\
                        help="acutance on the RGB value")
    parser.add_argument("--contrast_reduction", action='store_true',\
                        help="contrast reduction on dead leaves images")
    parser.add_argument("--new_mtf", action='store_true',\
                        help="contrast reduction on dead leaves images")
    parser.add_argument("--natural_only", action='store_true',\
                        help="activates natural training only")
    parser.add_argument("--filenames", "--fn", type=str, nargs = '+', default=['datasets/h5files/train_imnat/','datasets/h5files/train_dl/','datasets/h5files/val'], \
                        help="How many times to perform data augmentation")

    parser.add_argument("--lambda_acutance","--l_ac",type=float, default=10, 	\
                     help="Training batch size")
    # Saving parameters
    parser.add_argument("--check_point", "--cp", type=str, default = "ckpt.pth",\
                        help="resume training from a previous checkpoint")
    parser.add_argument("--save_every", type=int, default=10,\
                        help="Number of training steps to log psnr and perform \
                        orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=50,\
                        help="Number of training epochs to save state")
    parser.add_argument("--log_dir", type=str, default="logs", \
                     help='path of log files')
    parser.add_argument("--mode_csf", type=str, default="original", \
                     help='path of log files')
    argspar = parser.parse_args()
    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noiseIntL[0] /= 255.
    argspar.noiseIntL[1] /= 255.

    print("\n### Training FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(argspar)

