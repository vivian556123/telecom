#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
#%%

#%% SECTION 1 inclusion de packages externes 


from math import gcd
from re import M
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.segmentation import checkerboard_level_set
from skimage.segmentation import circle_level_set

#%% SECTION 2 - Lecture de l'image

im=skio.imread('coeurIRM.bmp')


#im=skio.imread('retineOA.bmp')

#im=skio.imread('brain.bmp')
#im=im[:,:,1]

#im=skio.imread('brain2.bmp')

plt.imshow(im, cmap="gray")

#%% SECTION 3a - Segmentation by contours actifs
im=skio.imread('brain.bmp')
im=im[:,:,1]

#im=skio.imread('coeurIRM.bmp')

s = np.linspace(0, 2*np.pi, 100)
init_x = 220
init_y = 220
rayon = 200
r = init_x + rayon*np.sin(s)
c = init_y + rayon*np.cos(s)
init = np.array([r, c]).T
a= 0.005 #a代表了tangent
b = 10 #beta代表了courbure b越小说明courbure可以越大，因为我们实际上最终是要minimiser总的能量
#w_l = 1
w_e = 1 #用于处理image
g = 0.01 #stepwise, 是inertie
bc = 'fixed'

snake = active_contour(gaussian(im, 0.1),
    init, alpha=a, beta=b, w_edge=w_e, gamma=g)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])
title = 'brain_ext/init'+str(init_x)+'_'+str(init_y)+'alpha'+str(a)+'beta'+str(b)+'w_edge'+str(w_e)+'gamma'+str(g)+'.png'
plt.savefig(str(title))
plt.show()




#%%W_edge
im=skio.imread('brain.bmp')
im=im[:,:,1]

#im=skio.imread('coeurIRM.bmp')

s = np.linspace(0, 2*np.pi, 100)
init_x = 120
init_y = 130
rayon = 20
r = init_x + rayon*np.sin(s)
c = init_y + rayon*np.cos(s)
init = np.array([r, c]).T
a= 0.1 #a代表了tangent
b = 1 #beta代表了courbure b越小说明courbure可以越大，因为我们实际上最终是要minimiser总的能量
w_e = -100 #用于处理image,越大表示蓝色离梯度很大的地方越近
g = 0.001 #stepwise
bc = 'fixed'

snake = active_contour(gaussian(im, 0.1),
    init, alpha=a, beta=b, w_edge=w_e, gamma=g)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])
title = 'brain_ext/init'+str(init_x)+'_'+str(init_y)+'alpha'+str(a)+'beta'+str(b)+'w_edge'+str(w_e)+'gamma'+str(g)+'.png'
plt.savefig(str(title))
plt.show()


#%% SECTION 3b - Contours ouverts

# Interessant sur l'image retineOA.bmp
im=skio.imread('retineOA.bmp')

r = np.linspace(60, 160, 100)
c = np.linspace(60, 200, 100)
init = np.array([r, c]).T
a= 0.01
b = 0.1
w_l = -10
w_e = 1
g=0.01
bc = 'free'

snake = active_contour(gaussian(im, 1), init,boundary_condition=bc,
                       alpha=a, beta=b, w_line=w_l, w_edge=w_e, gamma=g)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])
title = 'retine/alpha'+str(a)+'beta'+str(b)+'w_line'+str(w_l)+'w_edge'+str(w_e)+'gamma'+str(g)+'.png'
plt.savefig(str(title))
plt.show()

#%% SECTION 4 - Segmentation par ensembles de niveaux
im=skio.imread('brain.bmp')
im=im[:,:,1]
image = img_as_float(im)

# Init avec un damier
#init_ls = checkerboard_level_set(image.shape, 6)

# Init avec un cercle
#init_ls = circle_level_set (image.shape, (120,130), 10)
init_ls = circle_level_set (image.shape, (220,220), 40)

# Init avec plusieurs cercles
# circleNum = 8
# circleRadius = image.shape[0] / (3*circleNum)
# circleStep0 = image.shape[0]/(circleNum+1)
# circleStep1 = image.shape[1]/(circleNum+1)
# init_ls = np.zeros(image.shape)
# for i in range(circleNum):
#         for j in range(circleNum):
#             init_ls = init_ls + circle_level_set (image.shape, 
#                                                   ((i+1)*circleStep0, (j+1)*circleStep1), circleRadius)

m = 0.25
l1 = 5
l2 = 1
t = 1e-3
mt = 200
dt = 0.5


##parameter pre
#m=0.9
#l1=100
#l2=1
#t=1e-3
#mt=200

cv = chan_vese(image, mu=m, lambda1=l1, lambda2=l2, tol=t, max_iter=mt,
               dt=dt, init_level_set=init_ls, extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()

title = 'brain_chan_vese/Circlenum'+str(circleNum)+'mu'+str(m)+'lambda1'+str(l1)+'lambda2'+str(l2)+'tol'+str(t)+'maxiter'+str(mt)+'dt'+str(dt)+'.png'
plt.savefig(str(title))
plt.show()

     
#%% FIN  TP - Modeles deformables

# %%
