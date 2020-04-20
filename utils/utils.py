import torch


def one_hot_encode(categories, num_categories):
	'''
	Given a N sized vector describing the categories.

	Inputs:
		categories (tensor.Long): N-sized tensor describing the categories.
		num_categories (int): The number of categories.

	Returns:
		(tensor): N x M sized tensor where M is the max category. 
	'''

	one_hot_encodings = torch.zeros(categories.shape[0], num_categories)

	# One hot encode the visible ratings matrix.
	one_hot_encodings = one_hot_encodings.scatter(1, \
		torch.unsqueeze(categories, dim=1), 1)

	return one_hot_encodings


def create_visible_binary_indicator_matrix(visible_user_ratings_batch, \
	max_items, max_ratings):
	'''
	Creates the visible binary indicator matrix from the given users' ratings.

	Inputs:
		visible_user_ratings_batch (VisibleUserRatings): The users' ratings.
		max_ratings (int): The number of ratings (categories) = K
		max_items (int): The number of items = M

	Returns:
		(tensor): BxMxK where B is the number of users given.
	'''

	check_visible_user_ratings_ok(visible_user_ratings_batch)

	# Rename to make it easier. 
	B = len(visible_user_ratings_batch.ratings)
	M = max_items
	K = max_ratings

	V = torch.zeros(B, M, K)

	# One-hot encode each example in the batch.
	for i in range(B):
		ratings, item_indices = visible_user_ratings_batch.ratings[i], \
			visible_user_ratings_batch.item_indices[i]

		V[i][item_indices, ratings] = 1.0

	return V


def mask_missing_items(V, non_missing_item_indices):
	'''
	Given a visible binary matrix indicator that is BxMxK where:
		- B is the batch size
		- M is the item size
		- K is the rating category
	Where are K row vector represents the visible softmax units for item m, then
	masks all item indices per user that are not part of the 
	non_missing_item_indices.

	Inputs:
		V (tensor): BxMxK matrix describing batch of user visible softmax units.
		non_missing_item_indices (List[List[int]]): Item indices per user 
		 (M sublists).

	Returns:
		BxMxK

	This requires me to think of a fast way to compute batched MSE loss for visible units...
	'''


def check_visible_user_ratings_ok(visible_user_ratings):
	'''
	Given a VisibleUserRatings that describes a batch of user ratings on items, 
	asserts that it is valid. 

	Inputs:
		visible_user_ratings (VisibleUserRatings): The users' ratings.

	Throws:
		(ValueError): If it is invalid
	'''

	if len(visible_user_ratings.ratings) != \
		len(visible_user_ratings.item_indices):
		raise ValueError("Number of rating examples in VisibleUserRatings " \
			"must match number of item indices examples. {} != {}".format(
				len(visible_user_ratings.ratings), 
				len(visible_user_ratings.item_indices)))

