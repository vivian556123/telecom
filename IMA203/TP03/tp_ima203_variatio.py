#%%
# -*- coding: utf-8 -*-
import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio


# POUR LA MORPHO
from skimage.morphology import watershed 
from skimage.feature import peak_local_max

import time

#%%
# VOUS DEVEZ FIXER LES DEUX VARIABLES SUIVANTES: 
colaboratory=False #mettre True si vous utilisez google colab
notebook=False   # mettre Trus si vous utilisez un notebook local
# les seuls couples possibles sont (False,False)= travailler localement sans notebook
# (False,True): jupyternotebook local
# (True, False): google colab


assert (not (colaboratory and notebook)), "Erreur, choisissez google colab ou notebook local mais pas les deux en meme temps"

if colaboratory: #Si google colab on installe certaines librairies
    !pip install soundfile
    from IPython.display import Audio
    !pip install bokeh
    from bokeh.plotting import figure, output_file, show
    from bokeh.plotting import show as showbokeh
    from bokeh.io import output_notebook
    output_notebook()
    !wget https://perso.telecom-paristech.fr/ladjal/donnees_IMA203.tgz
    !tar xvzf donnees_IMA203.tgz
    os.chdir('donnees_IMA203')

if notebook: # si notebook normal dans une machine locale vous devez installer bokeh vous-meme
    from bokeh.plotting import figure, output_file, show
    from bokeh.plotting import show as showbokeh
    from bokeh.io import output_notebook
    output_notebook()
    



#%% fonction pour voir une image




def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0,titre=''):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    """
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a /Applications/GIMP.app '
        endphrase=' &' 
    elif platform.system()=='Linux': #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp -a '
        endphrase= ' &'
    elif platform.system()=='Windows':
        prephrase='start /B "D:/GIMP/bin/gimp-2.10.exe" -a '#Remplacer D:/... par le chemin de votre GIMP
        endphrase= ''
    else:
        print('Systeme non pris en charge par l affichage GIMP')
        return 'erreur d afficahge'
    """
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    
    if titre!='':
        titre='result/'+titre+'.png'
    else:
        titre='result/noname.png'
    #nomfichier=tempfile.mktemp('TPIMA'+titre+'.png')
    #commande=prephrase +nomfichier+endphrase
    #skio.imsave(nomfichier,imt)
    #os.system(commande)
    skio.imsave(titre,np.uint8(255*imt))

#si on est dans un notebook (y compris dans colab), on utilise bokeh pour visualiser

usebokeh= colaboratory or notebook
if usebokeh:
    def normalise_image_pour_bokeh(X,normalise,MINI,MAXI):
        imt=np.copy(X.copy())
        if normalise:
            m=imt.min()
            imt=imt-m
            M=imt.max()
            if M>0:
                imt=imt/M

        else:
            
            imt=(imt-MINI)/(MAXI-MINI)
            imt[imt<0]=0
            imt[imt>1]=1
        imt*=255
      
        sortie=np.empty((*imt.shape,4),dtype=np.uint8)
        for k in range(3):
            sortie[:,:,k]=imt
        sortie[:,:,3]=255
        return sortie
    def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0,titre=''):
        
        img=normalise_image_pour_bokeh(np.flipud(im),normalise,MINI,MAXI)# np.flipud(np.fliplr(im)))
        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],title=titre)
        p.x_range.range_padding = p.y_range.range_padding = 0

        # must give a vector of images
        p.image_rgba(image=[img], x=0,y=0, dw=im.shape[1], dh=im.shape[0])
        showbokeh(p)

  
#%% fonctions utiles au TP

def appfiltre(u,K):
    """ applique un filtre lineaire (en utilisant une multiplication en Fourier) """

    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    out=np.real(ifft2(fft2(u)*fft2(K)))
    return out    

def degrade_image(im,br): 
    """degrade une image en lui ajoutant du bruit"""
    out=im+br*np.random.randn(*im.shape)
    return out

def  grady(I):
    """ Calcule le gradient en y de l'image I, avec condition de vonnewman au bord
     i.e. l'image est symétrisée et le gradient en bas est nul"""
    
    (m,n)=I.shape
    M=np.zeros((m,n))
    M[:-1,:]=-I[:-1,:]+I[1:,:]
    M[-1,:]=np.zeros((n,))
    return M

def  gradx(I):
    """ Calcule le gradient en y de l'image I, avec condition de vonnewman au bord
     i.e. l'image est symétrisée et le gradient a droite est nul"""
    
    (m,n)=I.shape
    M=np.zeros((m,n))
    M[:,:-1]=-I[:,:-1]+I[:,1:]
    M[:,-1]=np.zeros((m,))
    return M

def div(px,py): 
    """calcule la divergence d'un champ de gradient"""
    """ div= - (grad)^*, i.e. div est la transposee de l'operateur gradient"""
    (m,n)=px.shape 
    assert px.shape==py.shape , " px et py n'ont pas la meme taille dans div"
    Mx=np.zeros((m,n))
    My=np.zeros((m,n))
    
    My[1:-1,:]=py[1:-1,:]-py[:-2,:]
    My[0,:]=py[0,:]
    My[-1,:]=-py[-2,:]
    
    Mx[:,1:-1]=px[:,1:-1]-px[:,:-2]
    Mx[:,0]=px[:,0]
    Mx[:,-1]=-px[:,-2]
    return Mx+My

def gradient_TV(v,u,lamb):
    """ calcule le gradient de la fonctionnelle E2 du TP"""
# on n'utilise pas gradx et grady car pour minimiser 
# la fonctionnelle E2 par descente de gradient nous avons choisi 
# de prendre les memes conditions au bords que pour la resolution quadratique
    (sy,sx)=v.shape
    Kx=np.zeros((sy,sx))
    Ky=np.zeros((sy,sx))
    Kx[0,0]=1
    Kx[0,1]=-1
    Ky[0,0]=1
    Ky[1,0]=-1
    Kxback=np.zeros((sy,sx))
    Kyback=np.zeros((sy,sx))
    Kxback[0,0]=-1
    Kxback[0,-1]=1
    Kyback[0,0]=-1
    Kyback[-1,0]=1

    Dx=appfiltre(u,Kx)
    Dy=appfiltre(u,Ky)
    ng=(Dx**2+Dy**2)**0.5+1e-5
    div=appfiltre(Dx/ng,Kxback)+appfiltre(Dy/ng,Kyback)
    return 2*(u-v)-lamb*div

def gradient_TV_nonperiodique(v,u,lamb):
     """ calcule le gradient de la fonctionnelle E2 du TP"""
     gx=gradx(u)
     gy=grady(u)
     ng=((gx**2)+(gy**2))**0.5+1e-5
     dive=div(gx/ng,gy/ng)
     return 2*(u-v)-lamb*dive
     

def resoud_quad_fourier(K,V):
    """trouve une image im qui minimise sum_i || K_i conv im - V_i||^2
     ou les K_i et les Vi sont des filtres et des images respectivement """
     
    n=len(K)
    assert len(K) == len(V) , "probleme de nombre de composantes dans resoud_quad"
    (sy,sx)=K[0].shape
    numer=np.vectorize(np.complex)(np.zeros((sy,sx)))
    denom=np.vectorize(np.complex)(np.zeros((sy,sx)))
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    for k in range(n):
        fV=fft2(V[k])
        fK=fft2(K[k])
        #print('type de fV',fV.dtype,' type de fK',fK.dtype)
        numer+=np.conj(fK)*fV
        denom+=abs(fK)**2
    return np.real(ifft2(numer/denom))

def minimisation_quadratique(v,lamb):
    """ minimise la fonctionnelle E1 du TP"""
    (sy,sx)=v.shape
    Kx=np.zeros((sy,sx))
    Ky=np.zeros((sy,sx))
    Kx[0,0]=1
    Kx[0,1]=-1
    Ky[0,0]=1
    Ky[1,0]=-1
    delta=np.zeros((sy,sx))
    delta[0,0]=1.0
    s=lamb**0.5
    K=(s*Kx,s*Ky,delta)#分mu
    V=(np.zeros((sy,sx)),np.zeros((sy,sx)),v) #分zi，0的fft是0
    return resoud_quad_fourier(K,V)

def norme_VT(I): 
    """ renvoie la norme de variation totale de I"""
    (sy,sx)=I.shape
    Kx=np.zeros((sy,sx))
    Ky=np.zeros((sy,sx))
    Kx[0,0]=1
    Kx[0,1]=-1
    Ky[0,0]=1
    Ky[1,0]=-1
    Dx=appfiltre(I,Kx)
    Dy=appfiltre(I,Ky)
    ng=(Dx**2+Dy**2)**0.5

def norme_VT_nonperiodique(u):
    gx=gradx(u)
    gy=grady(u)
    ng=((gx**2)+(gy**2))**0.5
    return ng.sum()

def norm2(x):
    return ((x**2).sum())**0.5

def E2_nonperiodique(u,v,lamb): # renvoie l'énergie E2
    return lamb*norme_VT_nonperiodique(u)+norm2(u-v)**2


def minimise_TV_gradient(v,lamb,pas,nbpas):
    """ minimise E2 par descente de gradient a pas constant """
    u=np.zeros(v.shape)
    Energ=np.zeros(nbpas)
    for k in range(nbpas):
        print(k)
        Energ[k]=E2_nonperiodique(u,v,lamb)
        u=u-pas*gradient_TV_nonperiodique(v,u,lamb)
    return (u,Energ)


def projection(I,a,itmax): 
    """ calcule la projection de I sur G_a
        G_a est le sous-gradient de TV en zero
        Comme vu dans le poly cette projection permet de resoudre le probleme
        de debruitage TV (E2)"""
    # ici on utilise les conditions au bord de von neuman 
    # i.e. on utilise gradx et grady definis plus haut et non pas une convolution circulaire
    (m,n)=I.shape
    t=0.1249
    px=np.zeros((m,n))
    py=np.zeros((m,n))
    un=np.ones((m,n))
    
    for it in range(itmax):
        N=div(px,py)-I/a
        Gx=gradx(N)
        Gy=grady(N)
        G=(Gx**2+Gy**2)**0.5
        pxnew=(px+t*Gx)/(un+t*G)
        pynew=(py+t*Gy)/(un+t*G)
        px=pxnew
        py=pynew
    # la projection est la divergence du champ px,py
    P=a*div(px,py)
    return P                

def vartotale_Chambolle(v,lamb,itmax=100):
    """ Trouve une image qui minimise lamb*TV(I)+||I-v||^2 
    en utilisant la projection dur G_a"""
    (m,n)=v.shape
    P=projection(v,lamb/2,itmax)
    print(norm2(P))
    return v-P


def imread(fichier):
    return np.float32(skio.imread(fichier))
#%% lire une image

im=imread('lena.tif') #ATTENTION IL FAUT ETRE DANS LE BON REPERTOIRE (utiliser os.chdir())


#%%

# voir l'image
viewimage(im,titre='ORIGINALE')
#degrader une image

imb=degrade_image(im,25)

# voir l'image bruitée 
viewimage(imb,titre='BRUITEE')

#%% restauration quadratique : exemple
lamb=2.5
restau=minimisation_quadratique(imb,lamb)
viewimage(restau,titre='RESTQUAD_LAMB='+str(lamb))

#%% dichotomie

def dichotomie (im,imb,lambda_max,lambda_min,epsilon):
    distance = np.linalg.norm(im-imb)
    utild = minimisation_quadratique(imb,lambda_max)
    while np.linalg.norm(utild - imb)**2 - distance**2 > epsilon:
        lambda_max = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)

    utild = minimisation_quadratique(imb,lambda_min)
    while np.linalg.norm(utild - imb)**2 - distance**2 <-epsilon:
        lambda_min = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)
        if lambda_max - lambda_min < 0.01:
            return (lambda_max + lambda_min )/2

    return (lambda_max + lambda_min )/2

lambda_tild = dichotomie(im,imb,20,0,0.1)
print(lambda_tild)
restau=minimisation_quadratique(imb,lambda_tild)
viewimage(restau,titre='RESTQUAD_LAMB='+str(lambda_tild))
print(norm2(im-restau))

#%%

def dichotomie_range (im,imb,lambda_max,lambda_min,epsilon):
    distance = np.linalg.norm(im-imb)
    utild = minimisation_quadratique(imb,lambda_max)
    while np.linalg.norm(utild - imb)**2 - distance**2 > epsilon:
        lambda_max = (lambda_max + lambda_min )/2
        print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)

    utild = minimisation_quadratique(imb,lambda_min)
    while np.linalg.norm(utild - imb)**2 - distance**2 <-epsilon:
        lambda_min = (lambda_max + lambda_min )/2
        print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)
        if lambda_max - lambda_min < 0.5:
            return (lambda_min,lambda_max)

    return (lambda_min,lambda_max)

#%%
def mini_lambda(im,imb):
    erreur=[]
    lamb_list = np.arange(0,0.2,0.01)
    for lamb in lamb_list:
        restq=minimisation_quadratique(imb,lamb)
        erreur.append(norm2(im-restq))
    index = np.argmin(erreur)
    print(index)
    print(lamb_list[index])
    #plt.plot(lamb_list, erreur)
    #plt.savefig('result/norm2(utild-u).png')
    return lamb_list[index]

mini_lambda(im,imb)
lambda_tild = 0.12
restau=minimisation_quadratique(imb,lambda_tild)
viewimage(restau,titre='RESTQUAD_LAMB='+str(lambda_tild))
print(norm2(im-restau))



#%% COMPARAISON des methodes
# vous pouvez vous inspirer de ce qui suit pour trouver les meilleurs 
# parametres de regularisation 

errq=[]
errvt=[]
erreur=[]
vk=np.arange(0,0.25,0.01)
for k in vk: 
    print (k)
    #restq=minimisation_quadratique(imb,10^(k))
    #errq.append[]=norm2(restq-myim)
    #restva=vartotale_Chambolle(imb,10**(k))
    restq=minimisation_quadratique(imb,10**(k))
    erreur.append(norm2(im-restq))
    #errvt.append(norm2(restva-myim));
    


plt.plot(10**vk,erreur)


#%%
#minimise_TV_gradient(v,lamb,pas,nbpas):

u1,en1=minimise_TV_gradient(imb, 40, 1, 20)
print("energie = {}".format(en1))
viewimage(u1,titre='minimise_TV_gradient_pas1')

u05,en05=minimise_TV_gradient(imb, 40, 0.5, 20)
viewimage(u05,titre='minimise_TV_gradient_pas0.5')
print("energie = {}".format(en05))

u01,en01=minimise_TV_gradient(imb, 40, 0.1, 20)
viewimage(u01,titre='minimise_TV_gradient_pas0.1')
print("energie = {}".format(en01))

u05inf,en05inf=minimise_TV_gradient(imb, 10, 0.5, 200)
viewimage(u05inf,titre='minimise_TV_gradient_pas0.5_inf')
print("energie = {}".format(en05inf))

#%%
#plt.plot(en1, label = 'pas = 1')
plt.plot(en05, label = 'pas = 0.5')
plt.plot(en01, label = 'pas = 0.1')
plt.legend()
plt.show()


#%%
u01,en01=minimise_TV_gradient(imb, 10, 0.1, 20)
#viewimage(u01inf,titre='minimise_TV_gradient_pas0.1_inf')
#print("energie = {}".format(en01inf))


#%%
print(norm2(u01inf - im))

#%%
#vartotale_Chambolle(v,lamb,itmax=100)

def mini_lambda_chambolle(im,imb):
    erreur=[]
    lamb_list = np.arange(0.0,60.0,0.1)
    for lamb in lamb_list:
        restq=vartotale_Chambolle(imb,lamb,30)
        erreur.append(norm2(im-restq))
    index = np.argmin(erreur)
    print(index)
    plt.plot(lamb_list, erreur)
    plt.savefig('result/chambolle_norm2(utild-u).png')
    return lamb_list[index]

#%%
#lamb = mini_lambda_chambolle(im,imb)
lamb = 0.12
itmax = 30
start = time.time()
chambolle = vartotale_Chambolle(imb,lamb,itmax)
end = time.time()
encham05 = E2_nonperiodique(chambolle,imb,lamb)
print("norm2:{}".format(norm2(im-chambolle)))
print("energie_cham05:{}".format(encham05))
print("time:{}".format(start-end))
#viewimage(chambolle,titre='Chambolle_'+str(lamb)+'iter_'+str(itmax))
#print(norm2(chambolle - im))



start = time.time()
u01,en01=minimise_TV_gradient(imb, lamb, 0.1, 100)
end = time.time()
print("norm2:{}".format(norm2(im-u01)))
print("energie_01:{}".format(en01[-1]))
print("time:{}".format(start-end))


#%%
myim=imread('lena.tif')
imb=degrade_image(myim,25)
(u,energ)=minimise_TV_gradient(imb,1,0.1,100)   # pas = 0.1
(u,energ2)=minimise_TV_gradient(imb,1,1,100)   
(u,energ05)=minimise_TV_gradient(imb,1,0.5,100)     # pas = 1
#%%
plt.figure(num=3,figsize=(8,5))
plt.plot(energ,label = 'lamb0.1')
plt.plot(energ05,label = 'lamb0.5')
plt.plot(energ2,label = 'lamb1')
#plt.legend(handles=[l1,l2],labels=['lamb0.1','lamb1'],loc='best')
plt.legend()
plt.show()


# %%
myim=imread('lena.tif')
imb=degrade_image(myim,25)
#Dichotomie

def dichotomie (im,imb,lambda_max,lambda_min,epsilon):
    distance = np.linalg.norm(im-imb)
    utild = minimisation_quadratique(imb,lambda_max)
    while np.linalg.norm(utild - imb)**2 - distance**2 > epsilon:
        lambda_max = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)

    utild = minimisation_quadratique(imb,lambda_min)
    while np.linalg.norm(utild - imb)**2 - distance**2 <-epsilon:
        lambda_min = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = minimisation_quadratique(imb,lambda_max)
        if lambda_max - lambda_min < 0.01:
            return (lambda_max + lambda_min )/2

    return (lambda_max + lambda_min )/2

lambda_tild = dichotomie(im,imb,20,0,0.1)
print("lambda_regularition:{}".format(lambda_tild))
restau=minimisation_quadratique(imb,lambda_tild)
viewimage(restau,titre='RESTQUAD_LAMB='+str(lambda_tild))
print(norm2(restau-im))



# %%
#Chambolle
itmax = 100

def dichotomie_chambolle (im,imb,lambda_max,lambda_min,epsilon):
    distance = np.linalg.norm(im-imb)
    utild = vartotale_Chambolle(imb,lambda_max,itmax)
    while np.linalg.norm(utild - imb)**2 - distance**2 > epsilon:
        #print("We are here")
        lambda_max = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = vartotale_Chambolle(imb,lambda_max,itmax)

    utild = vartotale_Chambolle(imb,lambda_min,itmax)
    while np.linalg.norm(utild - imb)**2 - distance**2 <-epsilon:
        lambda_min = (lambda_max + lambda_min )/2
        #print("lambda_min = {}, lambda_max = {}".format(lambda_min, lambda_max))
        utild = vartotale_Chambolle(imb,lambda_max,itmax)
        if lambda_max - lambda_min < 0.01:
            return (lambda_max + lambda_min )/2

    return (lambda_max + lambda_min )/2

lamb = dichotomie_chambolle(im,imb,500,0,0.01)
print("lambdachambolle:{}".format(lamb))
chambolle = vartotale_Chambolle(imb,lamb,itmax)
encham01 = E2_nonperiodique(chambolle,imb,lamb)
#print(encham01)
viewimage(chambolle,titre='Chambolle_lamb'+str(lamb)+'iter_'+str(itmax))

print(norm2(chambolle - im))
# %%
# %%


lamb = mini_lambda_chambolle(im,imb)
# %%
lamb = 31.25
itmax = 30
chambolle = vartotale_Chambolle(imb,lamb,30)
print(norm2(chambolle-im))

# %%
