from datasets.mini_batch_dataset import MiniBatchDataset
from utils.visible_user_ratings import VisibleUserRatings


class LargeMovielensMiniBatchDataset(MiniBatchDataset):
	
	'''
	Interface for getting mini batches of training data from a large MovieLens
	source (when it is not desirable to load the entire dataset into memory).

	This makes some assumptions, namely that the feedback dataset is ordered by
	user_id so that we can efficiently generate batches. (Every user gets one
	training example). Note: MovieLens data files are typically randomly 
	ordered. 

	MovieLens data files contain feedback as tab-separated lines:
	user id | item id | rating | timestamp

	The item and rating fields are NOT zero-indexed.

	Yield batches as VisibleUserRatings where items and ratings are
	zero-indexed.
	'''

	def get_batch(self):
		batch_ratings = []
		batch_item_indices = []

		with open(self._data_fp, 'r') as fp:
			current_user_id = None
			current_user_ratings = []
			current_user_item_indices =[]

			for line in fp:
				user_id, item_id, rating, timestamp = line.split("\t")
				user_id = int(user_id)
				item_id = int(item_id)
				rating = int(rating)

				if current_user_id is None:
					current_user_id = user_id

				if user_id != current_user_id:
					# Push all the ratings and items into the batch.
					batch_ratings.append(current_user_ratings)
					batch_ratings.append(current_user_item_indices)

					# Reset.
					current_user_ratings = []
					current_user_item_indices = []

					# Set the new user id.
					current_user_id = user_id

				# Add this data into the batch.
				current_user_ratings.append(rating)
				current_user_item_indices.append(item_id)

				if len(batch_ratings) == self._batch_size:
					yield VisibleUserRatings(batch_ratings, batch_item_indices)

					# Reset.
					batch_ratings = []
					batch_item_indices = []