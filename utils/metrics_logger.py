import torch
import torch.nn.functional as F
import numpy as np
import logging

from . import metrics_factory, figures_factory


class Avg_meter:
	"""
		running average meter
	"""

	def __init__(self):
		self.values = []
		self.count = 0
		self.total = 0

	def update(self, value):
		self.values.append(value)
		self.count += 1
		self.total += value

	def reset(self):
		self.values = []
		self.count = 0
		self.total = 0

	@property
	def median(self):
		d = torch.tensor(self.values)
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(self.values, dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.values)

	@property
	def value(self):
		return self.values[-1]


class Tb_scalar(Avg_meter):
	"""
		wrapper of Avg_meter to add a method for easy tensorboard logs
	"""

	def __init__(self, TB_logger=None, tag="metric_tag", default_log_type='last'):
		super().__init__()
		self.logger = TB_logger  # tensorboard logger object
		self.tag = tag
		self.default_log_type = default_log_type
		self.updated_since_last_log = 0

	def change_tag(self, new_tag):
		self.tag = new_tag

	def update(self, value):
		super().update(value)
		self.updated_since_last_log = 1

	def log(self, log_type=None, time_step=None):
		"""
			log_type : None or one of avg meter properties, refer to avg_meter for details
			(if None will use default_log_type)
			time_step : x axis value of the TB plots if None will use the count value of the avg meter
			method to log the metric to tensorboard
		"""
		if (self.count == 0):  # no values stored, no need to log
			return
		else:
			if (time_step is None):
				time_step = self.count

			if (log_type is None):
				log_type = self.default_log_type
			if (log_type == "last"):
				val = self.value
			else:
				try:
					val = getattr(self, log_type)
				except:
					print("log_type %s requested not implemented, will use default" % (log_type))
					val = self.value
			if (self.logger is None):
				print(time_step, val)
			else:
				if(self.updated_since_last_log != 0):
					self.logger.log_scalar(self.tag, val, time_step)
					self.count_last_log = self.count


class Tb_scalar_list:
	def __init__(self, TB_logger=None, tag_list=None, default_log_type='last'):
		if tag_list is None:
			tag_list = ["each_metric_tag"]
		self.logger = TB_logger
		self.nb_scalars = len(tag_list)
		self.tags = tag_list
		self.default_log_type = default_log_type
		self.reset()

	def reset(self):
		self.values = [Tb_scalar(self.logger, self.tags[i], self.default_log_type) for i in range(self.nb_scalars)]

	def change_tag(self, position, new_tag):
		self.values[position].change_tag(new_tag)

	def update(self, list_of_values):
		for i in range(len(list_of_values)):
			self.values[i].update(list_of_values[i])

	def get_property_all(self, property_name):
		l = []
		for scalar in self.values:
			if(scalar.count!=0):#make sure scalar is not empty
				if (hasattr(scalar, property_name)):#check if scalar has the requested property
					l.append(getattr(scalar, property_name))
		return l

	def log(self, log_type="last", time_step=None):
		for scalar in self.values:
			scalar.log(log_type, time_step)


class Tb_figure:
	"""
		simple class to log easily matplotlib figures to tensorboard
	"""

	def __init__(self, TB_logger=None, tag="figure_tag"):
		self.fig = None
		self.count = 0
		self.logger = TB_logger  # tensorboard logger object
		self.tag = tag

	def change_tag(self, new_tag):
		self.tag = new_tag

	def reset(self):
		self.fig = None
		self.count = 0

	def update(self, fig):
		self.fig = fig
		self.count += 1

	def log(self, time_step=None):
		if (self.count == 0):  # no figure stored, no need to log
			return
		else:
			if (time_step is None):
				time_step = self.count
			self.logger.log_fig(self.tag, self.fig, time_step)


class IMetrics_logger:
	"""
		Big class computing all incremental metrics when receiving info from the main training loops
	"""

	def __init__(self, args, Tb_logger, classes_names=None):
		self.current_epoch = 0
		self.tot_epochs = 0
		self.epoch_last_eval = -1
		self.current_iStep = 0
		self.Tb_logger = Tb_logger

		self.init_task_size = args.init_cls
		self.iTasks_sizes = args.cls_per_increment
		self.nb_tasks = args.nb_tasks
		self.nb_classes_iSteps = np.array([i * self.iTasks_sizes for i in range(self.nb_tasks)]) + self.init_task_size
		self.classes_names = classes_names if classes_names is not None else np.arange(self.nb_classes_iSteps[-1])

		# metrics logged every epochs
		self.epoch_metrics = {
			"epoch_time": Tb_scalar(Tb_logger, "ETA/epochs_ETA"),
			"loss_tot": Tb_scalar(Tb_logger, "losses/loss_tot", 'avg'),

			"train_acc": Tb_scalar(Tb_logger, "epoch_accuracies/train_accuracy"),
			"test_acc": Tb_scalar(Tb_logger, "epoch_accuracies/test_accuracy"),
			"auxFc_acc" : Tb_scalar(Tb_logger, "epoch_accuracies/auxiliaryFc_accuracy"),
			"pastTask_acc": Tb_scalar(Tb_logger, "epoch_accuracies/pastTask_accuracy"),
			"newTask_acc": Tb_scalar(Tb_logger, "epoch_accuracies/newTask_accuracy"),
			"initTask_acc": Tb_scalar(Tb_logger, "epoch_accuracies/initTask_accuracy")
		}
		# metrics logged every iStep
		self.iStep_metrics = {
			"iStep_time": Tb_scalar(Tb_logger, "ETA/iStep_ETA"),

			"tasks_ForgNEM": Tb_scalar_list(Tb_logger, ["forgetting_rates_NEM/task" + str(i)
														+ "_NEM_forgetting" for i in range(self.nb_tasks)]),
			"tasks_Forg": Tb_scalar_list(Tb_logger, ["forgetting_rates/task" + str(i)
													 + "_forgetting" for i in range(self.nb_tasks)]),
			"avg_Forg": Tb_scalar(Tb_logger, "forgetting_rates/average_forgetting"),
			"avg_ForgNEM" : Tb_scalar(Tb_logger, "forgetting_rates_NEM/average_NEM_forgetting"),

			"tasks_IAccNEM": Tb_scalar_list(Tb_logger, ["tasks_NEM_accuracies/task" + str(i)
														+ "_NEM_accuracies" for i in range(self.nb_tasks)]),
			"tasks_IAcc": Tb_scalar_list(Tb_logger, ["tasks_accuracies/task"
													 + str(i) + "_accuracies" for i in range(self.nb_tasks)]),

			"avg_IAcc": Tb_scalar_list(Tb_logger, ["iStep_accuracies/average_top"+str(k)+"_IAccuracy" for k in [1, 5]]),
			"IAcc": Tb_scalar_list(Tb_logger, ["iStep_accuracies/top"+str(k)+"_IAccuracy" for k in [1, 5]]),
			"IAcc_old": Tb_scalar(Tb_logger, "iStep_accuracies/IAccuracy_oldC"),
			"IAcc_new": Tb_scalar(Tb_logger, "iStep_accuracies/IAccuracy_newC"),

			"IAccNem": Tb_scalar_list(Tb_logger, ["iStep_accuracies_NEM/top"+str(k)+"_IAccuracyNEM" for k in [1, 5]]),
			"IAccNem_old": Tb_scalar(Tb_logger, "iStep_accuracies_NEM/IAccuracyNEM_oldC"),
			"IAccNem_new": Tb_scalar(Tb_logger, "iStep_accuracies_NEM/IAccuracyNEM_newC"),

			"tasks_IAccRaw": Tb_scalar_list(Tb_logger, ["Stats_before_calibration/task" + str(i)
														+ "_accuracies_NoCalibration" for i in range(self.nb_tasks)]),
			"IAccRaw": Tb_scalar(Tb_logger, "Stats_before_calibration/IAccuracy"),
			"IAccRaw_old": Tb_scalar(Tb_logger, "Stats_before_calibration/IAccuracy_oldC"),
			"IAccRaw_new": Tb_scalar(Tb_logger, "Stats_before_calibration/IAccuracy_newC"),
			"avg_ForgRaw": Tb_scalar(Tb_logger, "Stats_before_calibration/average_forgetting_NoCalibration"),

			#Tsne
			"tsne": Tb_figure(Tb_logger, "Tsne_visualisation"),
			"tsne_fewC": Tb_figure(Tb_logger, "Tsne_few_classes"),
			"bigModel_tsne": Tb_figure(Tb_logger, "Tsne_BigModel"),#(for CFECIL)
			"newFeats_tsne": Tb_figure(Tb_logger, "Tsne_NewFeats"),#(for CFECIL)

			#other figs
			"conf_mat": Tb_figure(Tb_logger, "confusion_matrix_cnn"),
			"k_neighbor_visu": Tb_figure(Tb_logger, "means_representativity"),
			"intraclass_variance": Tb_figure(Tb_logger, "intraclass_variance"),
			"features_norm" : Tb_figure(Tb_logger, "features_norm"),
			"means_distHeatmap": Tb_figure(Tb_logger, "interclassDist_heatmap"),
			"bigModel_meansHeatmap" : Tb_figure(Tb_logger, "bigModel_means_heatmap"),#(for CFECIL)
			"Wclassif_fig" : Tb_figure(Tb_logger, "classifier_weights"),
			"bigModel_Wclassif" : Tb_figure(Tb_logger, "BigModel_classifier_weights")
		}

	def info(self, key, *args, **kwargs):
		"""
			Main method to receive info from the learning process,
			key : string describing the input values
			kwargs : dict of values needed for computation of metrics
			(for example, predictions and labels for accuracy metrics)
		"""
		if (key == "train_loss"):
			self.update_loss_metrics(*args, **kwargs)
		elif (key == "epoch_eval"):
			self.update_epoch_metrics(*args, **kwargs)
		elif (key == "epoch_end"):
			self.epoch_log(*args, **kwargs)
		elif (key == "iStep_eval_noCalibration"):
			self.update_noCalibration_metrics(*args, **kwargs)
		elif(key == "classifier_calibration"):
			self.update_calibration_metrics(*args, **kwargs)
		elif (key == "iStep_eval"):
			self.update_iStep_metrics(*args, **kwargs)
		elif(key == "iStep_features"): #analyse feature space
			self.update_features_metrics(*args, **kwargs)
		elif (key == "iStep_end"):
			self.iStep_log(*args, **kwargs)

		else:
			logging.info("key " + str(key) + " not recognized, skipping metrics computations")

	def update_loss_metrics(self, *args, **kwargs):
		"""
			inputs : dict containing name of the loss and the loss tensor computed on current minibatch
		"""
		for lossName, lossVal in kwargs.items():
			if (lossName not in self.epoch_metrics):  # new loss (distillation for example), create category in dict
				self.epoch_metrics[lossName] = Tb_scalar(self.Tb_logger, "losses/" + str(lossName), 'avg')
			# update loss value
			self.epoch_metrics[lossName].update(lossVal.data)

	def update_epoch_metrics(self, preds_cnn, labels, preds_auxFc=None, **kwargs):
		"""
			preds_dict : dict containing all test predictions (topk_nem, topk_cnn) and labels
		"""
		self.epoch_last_eval = self.current_epoch
		top1_acc = metrics_factory.topk_accuracy(preds_cnn, labels, k=1)
		accInit = metrics_factory.init_task_acc(preds_cnn, labels, init_task_size=self.init_task_size, k=1)
		if(self.current_iStep>0):
			nbCOld = self.nb_classes_iSteps[self.current_iStep - 1]
			accOld, accNew = metrics_factory.newOld_accuracy(preds_cnn, labels, nbCOld, k=1)
		else:
			accOld = 0.
			accNew = top1_acc
		if(preds_auxFc is not None):
			top1_acc_aux = metrics_factory.topk_accuracy(preds_auxFc, labels, k=1)
			self.epoch_metrics['auxFc_acc'].update(top1_acc_aux)
		self.epoch_metrics['test_acc'].update(top1_acc)
		self.epoch_metrics['pastTask_acc'].update(accOld)
		self.epoch_metrics['newTask_acc'].update(accNew)
		self.epoch_metrics['initTask_acc'].update(accInit)

	def update_noCalibration_metrics(self, preds_cnn, labels, preds_nem=None):
		"""
			preds_cnn : dict containing cnn test predictions before the bias removal calibration step
			labels : True values of test set
		"""
		# compute tasks accs and forg before calibration
		# current tasks_accs
		tasks_accs_cnn = metrics_factory.tasks_topkAcc(preds_cnn, labels, self.init_task_size,
													   self.iTasks_sizes, self.current_iStep, k=1)
		self.iStep_metrics['tasks_IAccRaw'].update(tasks_accs_cnn)
		self.iStep_metrics['IAccRaw'].update(metrics_factory.topk_accuracy(preds_cnn, labels, k=1))
		if (self.current_iStep > 0):  # only during incremental steps
			nbCOld = self.nb_classes_iSteps[self.current_iStep - 1]
			accOld, accNew = metrics_factory.newOld_accuracy(preds_cnn, labels, nbCOld, k=1)
			self.iStep_metrics['IAccRaw_old'].update(accOld)
			self.iStep_metrics['IAccRaw_new'].update(accNew)
			# get all previous tasks_accs
			all_tasks_accs = self.iStep_metrics['tasks_IAcc'].get_property_all("values")
			for t_nbr in range(self.current_iStep):  # add current raw acc to prev accs for each task
				all_tasks_accs[t_nbr].append(tasks_accs_cnn[t_nbr])
			tasks_forg_cnn = metrics_factory.tasks_forgetting(all_tasks_accs)
			self.iStep_metrics['avg_ForgRaw'].update(np.mean(tasks_forg_cnn))

	def update_iStep_metrics(self, preds_cnn, labels, preds_nem=None):
		### MAIN METRICS
		top1_Iacc = metrics_factory.topk_accuracy(preds_cnn, labels, k=1)
		top5_Iacc = metrics_factory.topk_accuracy(preds_cnn, labels, k=5)
		self.iStep_metrics['IAcc'].update([top1_Iacc, top5_Iacc])
		avg_topk_IAccs = self.iStep_metrics['IAcc'].get_property_all('avg')
		self.iStep_metrics['avg_IAcc'].update(avg_topk_IAccs)
		tasks_accs_cnn = metrics_factory.tasks_topkAcc(preds_cnn, labels, self.init_task_size,
													   self.iTasks_sizes, self.current_iStep, k=1)
		self.iStep_metrics['tasks_IAcc'].update(tasks_accs_cnn)
		if(self.current_iStep>0):
			nbCOld = self.nb_classes_iSteps[self.current_iStep - 1]
			accOld, accNew = metrics_factory.newOld_accuracy(preds_cnn, labels, nbCOld, k=1)
			self.iStep_metrics['IAcc_old'].update(accOld)
			self.iStep_metrics['IAcc_new'].update(accNew)
			# get all tasks_accs
			all_tasks_accs = self.iStep_metrics['tasks_IAcc'].get_property_all("values")
			tasks_forg_cnn = metrics_factory.tasks_forgetting(all_tasks_accs)
			self.iStep_metrics['tasks_Forg'].update(tasks_forg_cnn)
			self.iStep_metrics['avg_Forg'].update(np.mean(tasks_forg_cnn[:-1]))#average forg of previous tasks

		#NEM METRICS
		if(preds_nem is not None):
			top1_IaccNem = metrics_factory.topk_accuracy(preds_nem, labels, k=1)
			top5_IaccNem = metrics_factory.topk_accuracy(preds_nem, labels, k=5)
			self.iStep_metrics['IAccNem'].update([top1_IaccNem, top5_IaccNem])
			tasks_accs_nem = metrics_factory.tasks_topkAcc(preds_nem, labels, self.init_task_size,
			                                               self.iTasks_sizes, self.current_iStep, k=1)
			self.iStep_metrics['tasks_IAccNEM'].update(tasks_accs_nem)
			if(self.current_iStep>0):
				accNemOld, accNemNew = metrics_factory.newOld_accuracy(preds_nem, labels, nbCOld, k=1)
				self.iStep_metrics['IAccNem_old'].update(accNemOld)
				self.iStep_metrics['IAccNem_new'].update(accNemNew)
				# get all tasks_accs
				all_tasks_accs_nem = self.iStep_metrics['tasks_IAccNEM'].get_property_all("values")
				tasks_forg_nem = metrics_factory.tasks_forgetting(all_tasks_accs_nem)
				self.iStep_metrics['tasks_ForgNEM'].update(tasks_forg_nem)
				self.iStep_metrics['avg_ForgNEM'].update(np.mean(tasks_forg_nem[:-1]))

		# OTHER METRICS
		conf_mat = figures_factory.conf_mat(preds_cnn, labels, log_scale=True)
		self.iStep_metrics['conf_mat'].update(conf_mat)

	def update_features_metrics(self, features, labels, means=None, feats_big=None, feats_new=None):

		nbCOld = 0 if self.current_iStep==0 else self.nb_classes_iSteps[self.current_iStep - 1]
		tsne_fig = figures_factory.tsne_fig(features, labels, nbCOld, verbose=1)
		self.iStep_metrics['tsne'].update(tsne_fig)

		#qualitative_tsne
		tsne_few_classes = figures_factory.qualitative_tsne(features, labels,
		                                        np.arange(self.init_task_size), self.classes_names, verbose=1)
		self.iStep_metrics['tsne_fewC'].update(tsne_few_classes)

		intraclass_var_fig = figures_factory.intraclass_variance_fig(features, labels)
		self.iStep_metrics['intraclass_variance'].update(intraclass_var_fig)
		feature_norms_fig = figures_factory.classes_feature_norms(features, labels)
		self.iStep_metrics['features_norm'].update(feature_norms_fig)
		if(means is not None):
			heatmap = figures_factory.means_dists_heatmap(means)
			self.iStep_metrics['means_distHeatmap'].update(heatmap)
			k_neighbor_fig = figures_factory.k_neighbors_visualization(features, labels, means)
			self.iStep_metrics['k_neighbor_visu'].update(k_neighbor_fig)
		if(feats_big is not None):
			tsne_fig = figures_factory.tsne_fig(feats_big, labels, nbCOld, verbose=1)
			self.iStep_metrics['bigModel_tsne'].update(tsne_fig)
			heatmap = figures_factory.means_dists_heatmap_from_feats(feats_big, labels)
			self.iStep_metrics['bigModel_meansHeatmap'].update(heatmap)
		if(feats_new is not None):
			tsne_fig = figures_factory.tsne_fig(feats_new, labels, nbCOld, verbose=1)
			self.iStep_metrics['newFeats_tsne'].update(tsne_fig)

	def epoch_log(self, prog_bar, train_acc):
		"""
			send end-of-epoch logs to tensorboard and reset metrics if needed
		"""
		self.epoch_metrics['train_acc'].update(train_acc)
		self.print_epochSummary(prog_bar)
		#log everything in tensorboard
		for key, metric in self.epoch_metrics.items():
			metric.log(time_step=self.tot_epochs)
			#reset losses for next epoch
			if('loss' in key):
				metric.reset()
		self.current_epoch += 1
		self.tot_epochs += 1

	def update_calibration_metrics(self, classifier=None, WA_coef=None, bigModel=False):
		if(classifier is not None):#compute classifier weights fig
			nbCOld = 0 if self.current_iStep==0 else self.nb_classes_iSteps[self.current_iStep - 1]
			nbCNew = self.nb_classes_iSteps[self.current_iStep] - nbCOld
			align_coef = None if WA_coef is None else WA_coef.cpu().numpy()
			w_fig = figures_factory.classif_weights_fig(classifier, align_coef, nbCOld, nbCNew)
			if(bigModel):
				self.iStep_metrics['bigModel_Wclassif'].update(w_fig)
				self.iStep_metrics['bigModel_Wclassif'].log(time_step=self.current_iStep)
			else:
				self.iStep_metrics['Wclassif_fig'].update(w_fig)

	def print_epochSummary(self, prog_bar):
		tot_epochs = prog_bar.total
		loss = self.epoch_metrics['loss_tot'].avg
		train_acc = self.epoch_metrics['train_acc'].value
		if(self.epoch_last_eval==self.current_epoch):#print eval
			test_acc = self.epoch_metrics['test_acc'].value
			accOld = self.epoch_metrics['pastTask_acc'].value
			accNew = self.epoch_metrics['newTask_acc'].value
			info = 'T {}, Ep {}/{} => L {:.3f}, TrainAcc {:.2f}, TestAcc {:.2f}|{:.2f}|{:.2f}'.format(
                self.current_iStep, self.current_epoch+1, tot_epochs, loss, train_acc, test_acc, accOld, accNew)
		else:#no eval
			info = 'T {}, Ep {}/{} => L {:.3f}, Train_acc {:.2f}'.format(
                self.current_iStep, self.current_epoch+1, tot_epochs, loss, train_acc)

		prog_bar.set_description(info)

	def iStep_log(self, *args):
		"""
			send end-of-iStep logs to tensorboard and reset metrics if needed
		"""
		self.print_iStepSummary()
		for key, metric in self.iStep_metrics.items():
			metric.log(time_step=self.current_iStep)

		self.current_epoch = 0
		self.current_iStep += 1
		pass

	def print_iStepSummary(self):
		topk_IAccs = self.iStep_metrics['IAcc'].get_property_all('value')
		topk_IAccsNEM = self.iStep_metrics['IAccNem'].get_property_all('value')
		if topk_IAccsNEM != []:
			logging.info('NEM Acc : top1 {}, top5 {}'.format(*topk_IAccsNEM))
		else:
			logging.info('No NME accuracy.')
		logging.info('CNN Acc : top1 {}, top5 {}'.format(*topk_IAccs))
		tasks_IAccs = self.iStep_metrics['tasks_IAcc'].get_property_all('value')
		tasks_IAccs_NEM = self.iStep_metrics['tasks_IAccNEM'].get_property_all('value')
		logging.info('Tasks CNN Accs : {}'.format(tasks_IAccs))
		if(tasks_IAccs_NEM != []):
			logging.info('Tasks NEM Accs : {}'.format(tasks_IAccs_NEM))

		topk_iAcc_curves = self.iStep_metrics['IAcc'].get_property_all('values')
		logging.info('CNN top1 Acc curve : {}'.format(topk_iAcc_curves[0]))
		logging.info('CNN top5 Acc curve : {}'.format(topk_iAcc_curves[1]))

