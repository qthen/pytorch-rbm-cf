from abc import ABC, abstractmethod


class MiniBatchDataset(object):
	
	'''
	Parent interface that any datasets yielding mini batches should inherit from.

	Every MiniBatchDataset child should own interpreting data for a particular 
	source (like Movielens or Netlfix) and be able to yield mini batches of 
	data. 

	Inputs:
		data_fp (str): The filepath for the data. 
		batch_size (int): The batch size to yield.

	Throws:
		(ValueError): On invalid params.
	'''
	def __init__(self, data_fp, batch_size):
		self._data_fp = data_fp
		self._batch_size = batch_size

		if not self._data_fp:
			raise ValueError("Data filepath cannot be empty.")

		if self._batch_size < 1:
			raise ValueError("Batch size must be positive.")


	@abstractmethod
	def get_batch(self):
		'''
		Generator for getting batches of the data, should yield it in a format 
		that the relevant model can intrepret (usually as VisibleUserRatings).

		Returns:
			(batch): Some batch.
		'''

		...