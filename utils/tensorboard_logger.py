import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Tensorboard_Logger():
	r"""
		simple logger class to record different metrics, images, matplotlib images, histograms, etc to tensorboard
	"""
	def __init__(self, record_folder, exp_name="testExperiment"):
		self.record_folder = record_folder #main tensorboard folder to store record files

		#use date to sort the experiment recordings
		#path will be record_folder/day/exp_name/hour
		date = datetime.now()
		day_str = date.strftime("%Y-%m-%d")
		hours_str = date.strftime("%Hh-%Mmin-%Ssec")

		#put time in record path to avoid overwriting recordings
		self.record_full_path = os.path.join(self.record_folder, day_str, exp_name, hours_str)

		self.writer = None #initialise on train start only to avoid creating useless folders when interrupting program before train start

	def setup_writer(self):
		print("tensorboard setup to write logs in :\n \t %s"%(self.record_full_path))
		self.writer = SummaryWriter(log_dir=self.record_full_path)

	def log_scalar(self, tag, value, step):
		self.writer.add_scalar(tag, value, step)

	def log_fig(self, tag, fig, step):
		self.writer.add_figure(tag, fig, step)

	def log_histo(self, tag, values, step, bins=1000):
		self.writer.add_histogram(tag, values, step, max_bins=bins)

	def log_img(self, tag, img_tensor, step, format="CHW"):
		'format represent the order of dimensions for the image (default is [Channels, Height, Width])'
		self.writer.add_image(tag, img_tensor, step, dataformats=format)