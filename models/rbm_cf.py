from abc import ABC, abstractmethod
from collections import namedtuple


class RbmCf(object):

	'''
	A parent interface for an RBM for collaborative filtering. Should always 
	have visible softmax units while the implementation of hidden units may differ.

	Inputs:
		M (int): The number of items (movies).
		K (int): The number of possible ratings for the items (movies).
		F (int): The number of hidden units.
	'''
	def __init__(self, M, K, F):
		self._M = M
		self._K = K
		self._F = F


	@abstractmethod
	def p_v_given_h(self, h, movie_indices):
		'''
		Probabilities of visible units given hidden: p(v_i^k = 1|h). 

		Inputs:
			h (tensor): The hidden vector.
			movie_indices (tensor): The indices of the movies we want to get 
			 the softmax visible units for (we typically do not want to
			 reconstruct all of them).
		Returns:
			(tensor): M x K matrix descirbing the M K-softmax units. 

		'''

		...


	@abstractmethod
	def p_h_given_v(self, ratings, movie_indices):
		'''
		Probabilities of hidden units given visible units: p(h_j=1|V). V is completely described by the ratings and the movie indices those ratings refer to.

		Inputs:
			ratings (tensor): M-sized tensor containing the user ratings.
			movie_indices (tensor): The indices of the movies these ratings refer to.
		'''

		...

	@abstractmethod
	def fit(self, visible_user_ratings_batch):
		'''
		Fit the model to a batch of user ratings, this method may be variadic
		so explicit parameters are left out of this declaration.

		Input(s):
			visible_user_ratings_batch (VisibleUserRatingsBatch): Batch of user
			 ratings on a set of items.
		'''

		...