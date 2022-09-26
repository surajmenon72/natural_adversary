import torch
import torchvision as tv
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

exp_dir = './checkpoints/adv_pths_knn_vicreg_base_fashion.pth'
save_dir = './checkpoints/adv_pths_knn_l_vicreg_base_fashion.pth'

knn_dict = torch.load(exp_dir)
knn_e = copy.deepcopy(knn_dict["knn_e"])
knn_t = copy.deepcopy(knn_dict["knn_t"])

train_dim_red = True

if (train_dim_red == True):
	print ('Doing PCA')
	knn_p = PCA(n_components=2).fit_transform(knn_e)
	print ('Done w/ PCA, PCA Shape:')
	print (knn_p.shape)

	print ('Doing T-SNE')
	knn_s = TSNE(n_components=2, learning_rate='auto',
	                  init='random').fit_transform(knn_e)
	print ('Done w/ T-SNE, T-SNE Shape')
	print (knn_s.shape)

	state = dict(
		knn_p = knn_p,
	    knn_s = knn_s,
	    knn_t = knn_t,
	)
	torch.save(state, save_dir)
else:
	knn_l_dict = torch.load(save_dir)
	knn_p = copy.deepcopy(knn_l_dict['knn_p'])
	knn_s = copy.deepcopy(knn_l_dict['knn_s'])
	print ('Loaded Dim Reduced Values')


N=10 #num classes
# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

plt.figure(1)
plt.scatter(knn_p[:, 0], knn_p[:, 1], c=knn_t, cmap=cmap)
plt.savefig('pca.png')

plt.figure(2)
plt.scatter(knn_s[:, 0], knn_s[:, 1], c=knn_t, cmap=cmap)
plt.savefig('tsne.png')