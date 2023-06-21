import numpy as np
import skimage



def hor_grad(c,n,angle):
    alpha = np.random.uniform(0,0.3)
    result = np.zeros((int(n+0.4*n),int(n+0.4*n),3))
    c = c/255.
    for k in range(int(n+0.4*n)):
        result[:,k,:] = (c-alpha) + (k/n)*(2*alpha)
    return(np.uint8(255*np.clip(skimage.transform.rotate(result,angle)[int(0.2*n):n+int(0.2*n),int(0.2*n):n+int(0.2*n)],0,1)))
def contrast_reduction(img):
    return(0.25+(img/255.)*0.5)
def f2(delta,r_min,r_max,width):
    d = 0
    for k in range(r_min,r_max,1):
        d+=(1/((1/r_min**2) -(1/r_max**2))*(1-((k/(k+1))**2)))
    d*=(np.pi)/(width**2)
    print(d)
    print(np.log(1-d))
    return(np.log(delta)/np.log(1-d))
def N_sup_r(r,width,r_min,r_max):
    delta = 100/(width**2)
    N = f2(delta,r_min,r_max,width)
    prop = ((1/r**2) - (1/r_max**2))/(1/(r_min**2) - 1/(r_max**2))
    res = N*prop
    return(res)