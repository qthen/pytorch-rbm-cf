from datasets.mini_batch_dataset import MiniBatchDataset
from utils.visible_user_ratings_batch import VisibleUserRatingsBatch


class MovielensMiniBatchDataset(MiniBatchDataset):
	
	'''
	Interface for getting mini batches of training data from a MovieLens 
	source that loads the entire dataset into memory. 

	Makes no assumptions about the structure or ordering of the data file.

	MovieLens data files contain feedback as tab-separated lines:
	user id | item id | rating | timestamp

	The item and rating fields are NOT zero-indexed.

	Yield batches as VisibleUserRatings where items and ratings are
	zero-indexed.
	'''
	def __init__(self, data_fp, batch_size):
		super().__init__(data_fp, batch_size)

		self._db = {} # Mappings of user_id -> ratings + item_indices
		with open(self._data_fp, 'r') as fp:
			for line in fp:
				user_id, item_id, rating, timestamp = line.split("\t")
				user_id = int(user_id)

				# Zero index item and rating.
				item_id = int(item_id) - 1
				rating = int(rating) - 1

				if user_id not in self._db:
					self._db[user_id] = {
						'ratings': [],
						'item_indices': []
					}

				self._db[user_id]['ratings'].append(rating)
				self._db[user_id]['item_indices'].append(item_id)


	def get_batch(self):
		batch_ratings = []
		batch_item_indices = []

		for user_id, user_information in self._db.items():
			batch_ratings.append(user_information['ratings'])
			batch_item_indices.append(user_information['item_indices'])

			if len(batch_ratings) == self._batch_size:
				yield VisibleUserRatingsBatch(batch_ratings, batch_item_indices)

				batch_ratings = []
				batch_item_indices = []