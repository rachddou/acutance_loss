import os
import argparse
from multiprocessing import Pool
import numpy as np 
from dead_leaves_generation.utils import *
from dead_leaves_generation.disk_dict import *
import numpy as np
from time import time, clock_gettime, CLOCK_MONOTONIC, sleep
import skimage.io as skio
import skimage
import argparse 
import os
from skimage.transform import pyramid_reduce
from skimage.filters import gaussian
from multiprocessing import Pool


# dict_instance = np.load('npy/dict.npy',allow_pickle=True)

def dead_leaves_image(alpha,img_source,mode,r_min,r_max,width,prog_blur):

    grad = True
    w,l = img_source.shape[0],img_source.shape[1]
    img = np.ones((width,width,3), dtype = np.uint8)
    if mode == "forward":
        binary_image = np.zeros((width+2*r_max+1,width+2*r_max+1))
    else :
        binary_image = np.ones((width,width), dtype = np.bool)
    k = 0
    # interval for the random radius
    vamin = 1/(r_max**(alpha-1))
    vamax = 1/(r_min**(alpha-1))
    n = width**2
    p = 100
    t0 = time()
    # add_rectangle = np.random.random()
    # add_rectangle = 1
    N_disk = f2(0.002,r_min,r_max,width)
    interval = int(N_disk/10)
    if interval == 0:
        interval = 1000
    while p >0.2:
        #get the random values and store them
        # defining the random radius
        r = vamin + (vamax-vamin)*np.random.random()
        r = int(1/(r**(1./(alpha-1))))
        color = np.uint8(img_source[np.random.randint(0,w),np.random.randint(0,l),:])
        pos = [np.random.randint(0,width),np.random.randint(0,width)]
        if mode == "backward":
            disk_mask_1d = binary_image[max(0,pos[0]-r):min(width,1+pos[0]+r),max(0,pos[1]-r):min(width,pos[1]+r+1)].copy()
        disk_1d = dict_instance[()][str(r)]
        if mode == "backward":
            disk_1d = disk_1d[max(0,r-pos[0]):min(2*r+1,width+r-pos[0]),max(0,r-pos[1]):min(2*r+1,width+r-pos[1])]
            disk_mask_1d *=  disk_1d
        else :
            disk_mask_1d =  np.float64(disk_1d)
        disk_mask= np.float32(np.repeat(disk_mask_1d[:, :, np.newaxis], 3, axis=2))
        # add the color value to the disk
        disk_mask_grad = disk_mask.copy()

        if r>30:
            angle = np.random.randint(0,360)
            if grad:
                d = hor_grad(color,2*r+1,angle)[0:disk_mask_grad.shape[0],0:disk_mask_grad.shape[1]]
                disk_mask_grad= np.uint8(np.float32(disk_mask_grad)*d)
            else : 
                disk_mask_grad = color*disk_mask_grad
        else :
            disk_mask_grad = color*disk_mask_grad
        
        # add the color value to the disk        
        # add the disk at the right place
        img[max(0,pos[0]-r):min(width,1+pos[0]+r),max(0,pos[1]-r):min(width,pos[1]+r+1),:]*=np.uint8(1-disk_mask)
        img[max(0,pos[0]-r):min(width,1+pos[0]+r),max(0,pos[1]-r):min(width,pos[1]+r+1),:]+=np.uint8(disk_mask_grad)

        if mode =="backward":
            binary_image[max(0,pos[0]-r):min(width,1+pos[0]+r),max(0,pos[1]-r):min(width,pos[1]+r+1)]*=np.logical_not(disk_mask_1d)
        elif mode =="forward":
            binary_image[max(0,pos[0]-r):min(width,1+pos[0]+r),max(0,pos[1]-r):min(width,pos[1]+r+1)]+= disk_mask_1d
        k+=1
            
        if k%interval ==0:
            print(time()-t0)
            print("number of disks : {:07d}".format(k))
            if mode =="backward":
                n = binary_image.sum()
            elif mode =="forward":
                n = (1-binary_image).sum()
            p = (100*n)/(width**2)
            print("percentage covered :{}%".format(p))
            if prog_blur :
                img = gaussian(img,2,multichannel=True)
    print(time()-t0)
    return(img)

def generation(args):
    np.random.seed()
    print("started")
    sleep(10)
    #loading params
    width = args.size
    r_min = args.rmin
    r_max = args.rmax
    direct = args.path
    if not os.path.exists(os.path.join(args.path_origin,direct)):
        os.makedirs(os.path.join(args.path_origin,direct))
    new_dir = os.path.join(args.path_origin,direct)
    if args.natural:
        path_imnat = os.path.abspath(args.color_path)
        print(path_imnat)
        files_orig = [os.path.join(path_imnat, f) for f in os.listdir(path_imnat) if os.path.isfile(os.path.join(path_imnat, f))]
        print(args.ind)
        files = [files_orig[i] for i in args.ind]
        N_imnat = len(files)
    for _ in range(args.number):
        blur = np.random.random()
        blur = 0
        if args.natural:
            ind = np.random.randint(0,N_imnat)
            img_source = skio.imread(files[ind])
            
        else :
            img_source = np.random.randint(0,255,(1000,1000,3))
        image = dead_leaves_image(args.alpha,img_source,args.mode,r_min,r_max,width,args.prog_blur)
        if img_source.shape[2]==3 :
            time_id = int(100*clock_gettime(CLOCK_MONOTONIC))
            if args.downscaling:
                img = pyramid_reduce(image,4, multichannel=True)
                if blur>0.9:
                    blur_value = np.random.uniform(1,3)
                    img = gaussian(img, blur_value)
                    img = np.clip(img,0,1)
                img = np.uint8(255*img)
                skio.imsave('{}/im_ds_{}.png'.format(new_dir,time_id), img)
            else:
                if args.grey :
                    image = skimage.color.rgb2gray(image)
                skio.imsave('{}/im_{}.png'.format(new_dir,time_id), image)

if __name__ == "__main__":

    if not os.path.isfile("npy/dict.npy"):
        disk_dict(1,2000)
    dict_instance = np.load('npy/dict.npy',allow_pickle=True)


    parser = argparse.ArgumentParser(description='Dead leaves image generator')
    ## general arguments
    parser.add_argument('--path'        , type=str  , default='datasets/dead_leaves_big_alpha_color_new_', 
                        metavar='P'     , help='path in which save the images')
    parser.add_argument('--mode'        , type=str  , default='backward', 
                        metavar='M'     , help='putting images in the front or in the back')
    parser.add_argument('--color_path'  , type=str, default="src_imgs/data", 
                        metavar='P'     ,help='path in which the color sources are')
    parser.add_argument('--path_origin' , type=str, default="datasets/", 
                        metavar='P'     ,help='path in which the color sources are')
    parser.add_argument('--size'        , type=int  , default=1000, 
                        metavar='S'     , help='size of the square side in the image')
    parser.add_argument('--rmin'        , type=int  , default=16, 
                        metavar='RMIN'  ,help='minimal size of the radius')
    parser.add_argument('--rmax'        , type=int  , default=1999, metavar='RMAX',
                        help='maximal size of the radius')
    parser.add_argument('--number'      , type=int  , default=1, 
                        metavar='N'     ,help='Number of image to generate with those parameters')
    parser.add_argument("--nb_p"        , type=int  , default= 30,
                        metavar='PR'    ,help='how many cores to parallelize on')
    parser.add_argument('--ind'         , type=int  ,nargs = '+'    , default=10, metavar='I',
                        help='which index to choose in')
    parser.add_argument('--grey'        , type=bool , default=False, 
                        metavar='C'     , help='True if grey, False if color')
    parser.add_argument('--alpha'       , type=float, default=3.0, 
                        metavar='A'     , help='exponent of the distribution of the radius')

    parser.add_argument('--downscaling' ,'--ds' , action='store_true',\
                        help="dowscales the image with a factor 5")
    parser.add_argument('--prog_blur'           , action ='store_true',\
                        help='True activates depth aware blur')
    parser.add_argument('--test'                , action ='store_true',\
                        help='creates a test set')
    parser.add_argument('--natural'             , action ='store_true',\
                        help='True if the color distribution is extracted from natural images/random otherwise')
    parser.add_argument('--partial_images'          , action ='store_true',\
                        help='true if all color images are present in the generation algo')

    args, unknown = parser.parse_known_args()
    path_imnat = os.path.abspath(args.color_path)
    N_images = len([os.path.join(path_imnat, f) for f in os.listdir(path_imnat) if os.path.isfile(os.path.join(path_imnat, f))])
    if args.partial_images:
        args.ind = list(np.random.choice(np.arange(0,N_images,1,dtype = np.uint16),100))
    else:
        args.ind = list(np.arange(0,N_images,1))
        
    #PATH = "Travail/these/ffdnet_raphael/dead_leaves/"
    pool = Pool(args.nb_p)
    # pool.map(beta_test,[ (1,2) for _ in range(10)])
    pool.map(generation,[args for i in range(args.nb_p)])
