import datasets
from base import BaseDataLoader

class HenryDataLoader(BaseDataLoader):

	def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):

		self.data_dir = data_dir
		self.dataset = datasets.HenryDataset

		super(HenryDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

	def __len__(self):

	def __getitem__(self):
