import numpy as np

def topk_accuracy(topk_pred, y_true, k=1):
	"""
		topk_pred : size [nb_data, nb_preds] (nb_preds>=k) or [nb_data,] (top1 acc)
		labels : size [nb_data,]
	"""
	nb_data = len(y_true)
	if(type(topk_pred) is not np.ndarray):
		topk_pred = np.array(topk_pred)
	if(type(y_true) is not np.ndarray):
		y_true = np.array(y_true)

	nb_preds = 1 if len(topk_pred.shape)<=1 else topk_pred.shape[1]
	topk_pred = topk_pred.reshape((nb_data, nb_preds))[:,:k]#normalize shapes to [nb_data, k] even if k=1

	topk_acc = np.around((topk_pred.T == np.tile(y_true, (k, 1))).sum()*100/len(y_true), decimals=2)
	if(np.isnan(topk_acc)):
		print(y_true)
	return topk_acc

def mask_accuracy(topk_pred, y_true, mask, k=1):
	"""
		compute accuracy on specific classes using model preds, labels, and a bool mask for those classes
	"""
	nb_data = len(y_true)
	if(type(topk_pred) is not np.ndarray):
		topk_pred = np.array(topk_pred)
	if(type(y_true) is not np.ndarray):
		y_true = np.array(y_true)

	nb_preds = 1 if len(topk_pred.shape)<=1 else topk_pred.shape[1]
	topk_pred = topk_pred.reshape((nb_data, nb_preds))[:,:k]#normalize shapes to [nb_data, k] even if k=1

	preds = topk_pred[mask, :]
	y_m = y_true[mask]
	topk_accMasked = topk_accuracy(preds, y_m, k)
	return topk_accMasked

def newOld_accuracy(topk_pred, y_true, nb_old, k=1):
	"""
		Computes topk accuracy on old classes and new classes separately
	"""
	if(type(y_true) is not np.ndarray):
		y_true = np.array(y_true)

	maskOld = y_true<nb_old
	topk_accOld = mask_accuracy(topk_pred, y_true, maskOld, k)
	topk_accNew = mask_accuracy(topk_pred, y_true, ~maskOld, k)
	return topk_accOld, topk_accNew

def init_task_acc(topk_pred, y_true, init_task_size, k=1):

	if(type(y_true) is not np.ndarray):
		y_true = np.array(y_true)
	maskInit = y_true<init_task_size
	topk_accInit = mask_accuracy(topk_pred, y_true, maskInit, k)
	return topk_accInit

def tasks_topkAcc(topk_pred, y_true, init_task_size, iTasks_sizes, current_iStep, k=1):
	"""
		computes accuracy on each task separately
	"""
	nb_data = len(y_true)
	if(type(topk_pred) is not np.ndarray):
		topk_pred = np.array(topk_pred)
	if(type(y_true) is not np.ndarray):
		y_true = np.array(y_true)

	nb_preds = 1 if len(topk_pred.shape)<=1 else topk_pred.shape[1]
	topk_pred = topk_pred.reshape((nb_data, nb_preds))[:,:k]#normalize shapes to [nb_data, k] even if k=1

	task_label_range = [0, init_task_size]
	tasks_accs = np.zeros((current_iStep+1,))#task 0 in init training
	for task_nbr in range(current_iStep+1):
		if(task_nbr>0):
			task_label_range = [task_label_range[1], task_label_range[1]+iTasks_sizes]

		taskC_mask = np.logical_and(y_true>=task_label_range[0], y_true<task_label_range[1])
		task_pred = topk_pred[taskC_mask, :]
		task_y = y_true[taskC_mask]

		topk_task_acc = topk_accuracy(task_pred, task_y, k)
		tasks_accs[task_nbr] = topk_task_acc
	return tasks_accs

def topk_forgetting(topk_iAccs):
	"""
		given the accuracy after each iStep, computes the forgetting at current iStep
		topk_iAccs : size [nb_iSteps,]
	"""
	max_acc = np.max(topk_iAccs)
	current_acc = topk_iAccs[-1]
	forgetting = max_acc - current_acc
	return forgetting

def tasks_forgetting(tasks_topkAcc):
	"""
		given all seen tasks accuracies after each iStep, compute each task forgetting
		tasks_accuracies : list of lists with accuracy of each tasks after each iStep
	"""
	tasks_forgetting = []
	for task_accs in tasks_topkAcc:
		task_f = topk_forgetting(task_accs)
		tasks_forgetting.append(task_f)
	return tasks_forgetting