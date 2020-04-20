from collections import namedtuple


'''
POD struct that describes a batch of "observed binary indicator matrix".

Inputs:
	ratings (List[List[int]]): Describes a batch of user ratings.
	itme_indices: (List[List[int]]): Describes a batch of item indices for the 
	 user ratings.
'''
VisibleUserRatingsBatch = namedtuple("VisibleUserRatingsBatch", ("ratings", \
	"item_indices"))