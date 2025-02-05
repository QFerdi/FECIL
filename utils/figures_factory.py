from sklearn.manifold import TSNE
from scipy.spatial import distance
import sklearn.metrics
import numpy as np
from matplotlib import cm
#import seaborn as sns
#sns.set()#set default figure params for better graphs
import matplotlib.pyplot as plt

def conf_mat(preds, labels, classes_names=None, log_scale=False):
	r"""
		uses sklearn to compute a confusion matrix
		log_scale : if true all entries are scaled to log(1+x) to increase visibility of lower values

		returns the matplotlib figure
	"""
	preds = np.array(preds)
	if(len(preds.shape)>1):#multiple preds, take only top1
		preds = preds[:,0]
	#
	cM = sklearn.metrics.confusion_matrix(labels, preds, labels=classes_names, normalize="true")
	# disp = sklearn.metrics.ConfusionMatrixDisplay(cM)
	# disp.plot()
	# fig = disp.figure_

	maxVal = 1.
	if(log_scale):#in some papers they apply log(1+x) to all entries for better visibility
		cM = np.log(1.+cM)
		maxVal = np.log(1.+maxVal)
	#tmp show plot
	fig, ax = plt.subplots()
	im1 = ax.imshow(cM, interpolation='none', vmin=0, vmax=maxVal)
	fig.colorbar(im1, ax=ax)

	nb_classes = cM.shape[0]
	ticks = np.linspace(0, nb_classes, min(20, nb_classes), endpoint=False, dtype=int)
	plt.xticks(ticks, labels=classes_names, fontsize=6, rotation=90)
	plt.xlabel("predictions")
	plt.yticks(ticks, labels=classes_names, fontsize=6)
	plt.ylabel("labels")
	plt.tight_layout()
	plt.grid(False)
	# plt.show()

	return fig

def classif_weights_fig(fc_layer, alignmentCoef=None, nbPastC=0, nbNewC=10):
	r"""
		shows the norm of the weight vectors of each class in the classifier and in different color for
		past classes and new classes to compare them. if alignmentCoef is given will also show the coef used in WA.
	"""
	w = fc_layer.weight.data.cpu().numpy()
	b = None
	if(fc_layer.bias is not None):
		b = fc_layer.bias.data.cpu().numpy()
	nbCTot = w.shape[0]
	fig, ax = plt.subplots()

	#weights
	w_norms = np.linalg.norm(w, ord=2, axis=1)

	newC = np.arange(nbPastC, nbPastC+nbNewC)
	norms_newC = w_norms[newC]
	ax.scatter(newC, norms_newC, marker='.', color="red", label="new classes weights")
	if(b is not None):
		ax.scatter(newC, b[newC], marker='x', color="red", label="new classes bias")
	if(nbPastC>0):
		pastC = np.arange(nbPastC)
		norms_pastC = w_norms[pastC]
		ax.scatter(pastC, norms_pastC, marker='.', color="blue", label="past classes weights")
		if(b is not None):
			ax.scatter(pastC, b[pastC], marker='x', color="blue", label="past classes bias")

	#classes not seen yet if weights preinitialized
	if(nbCTot>nbPastC+nbNewC):
		notSeenC = np.arange(nbPastC+nbNewC, nbCTot)
		norms_notseenC = w_norms[notSeenC]
		ax.scatter(notSeenC, norms_notseenC, marker='+', color='gray', label="unseen classes weights")

	#figure params
	if(alignmentCoef is not None):
		ax.set_title("new classes weights aligned with coef {:.3f}".format(alignmentCoef))
	ax.set_xlabel("classes indexes")
	ax.set_ylabel("classifier weight norms")
	ax.grid(True)
	ax.legend()
	return fig

def tsne_fig(features, labels, nbOld_classes, verbose=0):
	r"""
		A tsne will be trained to represent Nd features vectors in 2d.
		new classes will be in orange and past ones in blue
	"""
	print("computing tsne visualisation")
	mask_oldC = labels<nbOld_classes
	nb_data = features.shape[0]
	tsne = TSNE(n_components=2, init='pca', learning_rate=max(nb_data/48, 50), verbose=verbose)
	X = tsne.fit_transform(features)

	fig, ax = plt.subplots()
	plt.title('T-SNE visualisation of the feature space')
	ax.scatter(X[mask_oldC,0], X[mask_oldC,1], c="tab:blue", s=2)
	ax.scatter(X[~mask_oldC,0], X[~mask_oldC,1], c="tab:orange", s=2)
	ax.grid(False)
	return fig

def qualitative_tsne(features, labels, classes, classes_names, verbose=0):
	r"""
		a tsne will be trained to represent the specified classes feature vectors in 2d
	"""
	print("computing qualitative tsne visualisation")
	nb_data_tot = labels.shape[0]
	#mask out unwanted classes
	mask = np.zeros(nb_data_tot,)
	for c in classes:
		mask[labels==c] = 1
	f,l = features[mask.astype(bool)], labels[mask.astype(bool)]
	nb_data = l.shape[0]

	#train tsne
	tsne = TSNE(n_components=2, init='pca', learning_rate=max(nb_data/48, 50), verbose=verbose)
	X = tsne.fit_transform(f)

	#plot
	if(len(classes)<=12):
		colormap = cm.get_cmap('Set3')
		colors = iter([colormap(i) for i in range(len(classes))])
	else:
		colormap = cm.get_cmap('viridis')
		colors = iter(colormap(np.linspace(0,1, len(classes))))
	fig, ax = plt.subplots(figsize=(14,10))
	plt.title('T-SNE visualisation of the feature space')
	for i,c in enumerate(classes):
		x = X[l==c]
		ax.scatter(x[:,0], x[:,1], s=8, c=np.array([next(colors)]), label=classes_names[i])
	if(len(classes)<=12):
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
	ax.grid(False)
	return fig

def means_dists_heatmap(means, distMetric="euclidean"):
	#means : [nbC, nbFeatures]
	nbC = means.shape[0]
	if(distMetric=="euclidean"):
		Dmax = 2.
		D = np.zeros((nbC, nbC))
		for c in range(nbC):
			classMean = means[c,:]
			#euclidean distance
			D[c,:] = np.sqrt(np.sum((means - classMean)**2, 1))
	else:#cosine distance
		Dmax = 2.
		w = np.linalg.norm(means, ord=2, axis=1, keepdims=True)
		D = 1 - (np.matmul(means, np.transpose(means)) / (w * np.transpose(w)).clip(min=1e-8))

	fig = plt.figure()
	plt.title('classes means distances heatmap')
	plt.imshow(D, cmap='jet', origin='lower', vmin=0., vmax=Dmax)

	ticks = np.linspace(0, nbC, min(20, nbC), endpoint=False, dtype=int)
	plt.xticks(ticks)
	plt.yticks(ticks)
	plt.xlabel('classes')
	plt.ylabel('classes')
	plt.grid(False)

	plt.colorbar()
	return fig
def means_dists_heatmap_from_feats(feats, labels, distMetric="euclidean"):
	#feats : [nbData, nbFeatures]
	nb_features = feats.shape[1]
	labels = np.array(labels)
	classes = set(labels)
	nbC = len(classes)
	means = np.zeros((nbC,nb_features))
	for i in range(nbC):
		classMask = labels == i
		classData = feats[classMask, :]
		#norm = np.linalg.norm(classData, ord=2, axis=1, keepdims=True)
		mean = np.mean(classData, axis=0)
		means[i,:] = mean/np.linalg.norm(mean, ord=2)
	return means_dists_heatmap(means, distMetric)

def intraclass_variance_fig(features, labels):
	#normalize feats
	features = features / (np.linalg.norm(features, axis=1, keepdims=True))
	labels = np.array(labels)
	classes = set(labels)
	nbC = len(classes)
	intraClassVars = np.zeros((nbC,))
	for i in range(nbC):
		classMask = labels == i
		classData = features[classMask, :]
		intraClassVars[i] = np.mean(np.cov(classData, rowvar=False))
		#intraClassVars[i] = np.mean(np.var(classData, ddof=1, axis=0))#mean variance
		#intraClassVars[i] = np.mean(np.std(classData, axis=0))#mean std
	x = np.arange(nbC)
	fig, ax = plt.subplots()
	plt.title('avg intra class variance in feature space')
	plt.bar(x, intraClassVars)
	#ax.set_ylim(0, 1.)
	ax.set_ylabel('features mean covariance')
	ax.set_xlabel('classes')
	ticks = np.linspace(0, nbC, min(20, nbC), endpoint=False, dtype=int)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks)
	return fig

def classes_feature_norms(features, labels):
	labels = np.array(labels)
	classes = set(labels)
	nbC = len(classes)
	norms = np.zeros((nbC,))
	for i in range(nbC):
		classMask = labels == i
		classData = features[classMask, :]
		norms[i] = np.mean(np.linalg.norm(classData, ord=2, axis=1))
	x = np.arange(nbC)
	fig, ax = plt.subplots()
	plt.title('avg feature vector norm of each class')
	plt.bar(x, norms)
	#ax.set_ylim(0, 1.)
	ax.set_ylabel('features mean norm')
	ax.set_xlabel('classes')
	ticks = np.linspace(0, nbC, min(20, nbC), endpoint=False, dtype=int)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks)
	return fig


def classes_repartition_fig(nb_each_class_iStep, classes, nbPastC, nbNewC):
	r"""
		compute an histogram representing how much each class was represented overall in the target onehot vectors
		during the incremental step (ie. shows the dataset bias)
	"""
	probs = nb_each_class_iStep/sum(nb_each_class_iStep)*100
	fig, ax = plt.subplots()
	x = np.arange(len(classes))
	ax.bar(x, probs)
	ax.set_xticks(x)
	ax.set_xticklabels(classes, fontdict={'fontsize':8})
	plt.title("number of times each class appeared in the target onehot vectors")
	ax.set_ylabel("number of occurrences")
	ax.set_xlabel('classes')
	ax.grid(True)
	plt.tight_layout()
	return fig

def k_neighbors_visualization(feats, labels, means, K=500):
	r"""
		Computes the K nearest neighbors to each means
		and plot the percentage of these neighbors that come
		from the same class, to see how representative the means are
	"""
	#feats : [nbData, nbFeatures]
	#normalize feats
	feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True))
	nb_features = feats.shape[1]
	labels = np.array(labels)
	classes = set(labels)
	nbC = len(classes)
	means_knn_percentage = np.zeros((nbC,))
	D = distance.cdist(feats, means) #size [nbD, nbC]
	for c in range(nbC):
		dists = D[:,c]
		ind_k_smallest = np.argpartition(dists, K)[:K]
		label_inds = labels[ind_k_smallest]
		nb_same_class = np.sum(label_inds == c)
		means_knn_percentage[c] = nb_same_class/K * 100.0

	#fig params
	x = np.arange(nbC)
	fig, ax = plt.subplots()
	ax.bar(x, means_knn_percentage)

	#plt.title("%d nearest exemplars to each class mean feature vector" %(K))
	#ax.set_ylabel("percentage of nearest exemplars belonging to mean's class")
	plt.title("Classes mean representativity")
	ax.set_ylabel("percentage of accuracy")
	ax.set_xlabel('classes')
	ticks = np.linspace(0, nbC, min(20, nbC), endpoint=False, dtype=int)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks)
	# ax.set_xlim(0, 100)
	ax.set_ylim(0, 100)
	ax.grid(True)
	return fig