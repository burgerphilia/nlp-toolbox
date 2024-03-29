import os
import math
import json
import logging
import datetime
import torch

from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX

class BaseTrainer:

	def __init__(self, model, loss, metric, optimizer, resume, config, train_logger=None):

	self.config = config
	self.logger = logging.getLogger(self.__class__.__name__)		

	self.device, device_ids = self._prepare_device(config['n_gpu'])
	self.model = model.to(self.device)

	if len(device_ids) > 1:

		self.model = torch.nn.DataParallel(model, device_ids=device_ids)

	self.loss = loss
	self.metrics = metrics
	self.optimizer = optimizer
	self.train_logger = train_logger

	cfg_trainer = config['trainer']
	self.epochs = cfg_trainer['epochs']
	self.save_period = cfg_trainer['save_period']
	self.verbosity = cfg_trainer['verbosity']
	self.monitor = cfg_trainer.get('monitor', 'off')

	if self.monitor == 'off':

		self.mnt_mode = 'off'
		self.mnt_best = 0

	else:

		self.mnt_mode, self.mnt_metric = self.monitor.split()
		assert self.mnt_mode in ['min', 'max']

		self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
		self.early_stop = cfg_trainer.get('early_stop', math.inf)

	self.start_epoch = 1

	start_time = datatime.datatime.now().strftime('%m%d_%H%M%S')
	self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], start_time)

	writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'],start_time)
	self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])

	ensure_dir(self.checkpoint_dir)
	config_save_path = os.path.join(self.checkpoint_dir, 'config.json')

	with open(config_save_path, 'w') as handle:

		json.dump(config, handle, indent=4, sort_keys=False)

	if resume:

		self._resume_checkpoint(resume)


	def _prepare_device(self, n_gpu_use):

		n_gpu = torch.cuda.device_count()
		if n_gpu_use > 0 and n_gpu == 0:

			self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
			n_gpu_use = 0

		if n_gpu_use > n_gpu:

			self.logger.warning("Warning: The number of GPU\'s configured to use but only on this machine".format(n_gpu_use, n_gpu))
			n_gpu_use = n_gpu

		device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
		list_ids = list(range(n_gpu_use))

		return device, list_ids

	def train(self):

		for epoch in range(self.start_epoch, self.epochs + 1):

			result = self._train_epoch(epoch)

			log = {'epoch': epoch}

			for key, value in result.items():

				if key == 'metrics':
					log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'val_metrics':
					log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
				else:
					log[key] = value

			if self.train_logger is not None:
				self.train_logger.add_entry(log)
				if self.verbosity >= 1:
					for key, value in log.items():
						self.logger.info('    {:15s}: {}'.format(str(key), value))

			best = False
			if self.mnt_mode != 'off':
				try:
					improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)

				except KeyError:
					self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
					self.mnt_mode = 'off'
					improved = False
					not_improved_count = 0

				if improved:

					self.mnt_best = log[self.mnt_metric]
					not_improved_count = 0
					best = True

				else:

					not_imporved_count += 1

				if not_improved_count > self.early_stop:

					self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
					break

			if epoch % self.save_period == 0:

				self._save_checkpoint(epoch, save_best=best)
	
	def _train_epoch(self, epoch):

		raise NotImplementedError

	def _save_checkpoint(self, epoch, save_best=False):

		arch = type(self.model).__name__
		state = {
		'arch': arch, 
		'epoch': epoch, 
		'logger': self.train_logger, 
		'state_dict': self.model.state_dict(), 
		'monitor_best': self.mnt_best, 
		'config': self.config
		}

		filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
		torch.save(state, filename)
		self.logger.info("Saving checkpoint: {} ...".format(filename))
		if save_best:
			best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
			torch.save(state, best_path)				
			self.logger.info("Saving current best: {} ...".format('model_best.pth'))

	def _resume_checkpoint(self, resume_path):

		self.logger.info("Loading checkpoint: {} ...".format(resume_path))
		checkpoint = torch.load(resume_path)
		self.start_epoch = checkpoint['epoch'] + 1
		self.mnt_best = checkpoint['monitor_best']

		if checkpoit['config']['arch'] != self.config['arch']:

			self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.')

		self.model.load_state_dict(checkpoint['state_dict'])


		if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
			self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed')
		else:
			self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.train_logger = checkpoint['logger']
		self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))