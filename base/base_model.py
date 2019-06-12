import logging
import torch.nn as nn
import numpy as np

class BaseModel(nn.Moudule):

	def __init__(self):

		super(BaseModel, self).__init__()
		self.logger = logging.getLoger(self.__class__.__name__)

	def forward(self, *input):

		raise NotImplementedError

	def summary(self):

		model_parameters = filter(lambda parameter: parameter.requires_grad, self.parameters())
		params = sum([np.prod(parameter.size()) for parameter in model_parameters])

		self.logger.info('Trainable parameters: {}'.format(params))
		self.logger.info(self)

	def __str__(self):

		model_parameters = filter(lambda parameter: parameter.requires_grad, self.parameters())
		params = sum([np.prod(parameter.size()) for parameter in model_parameters])

		return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
		
