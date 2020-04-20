class UserFeedback(object):

	'''
	An (almost) POD struct of user feedback for a certian item. In general,
	UserFeedback must contain the following fields:
		- user_id
		- item_id (movie_id)
		- rating
		- timestamp

	Defines various factory methods for constructing a UserFeedback from a
	string from data sources.
	'''

	@staticmethod
	def create_from_movielens(tsl):
		'''
		Creates a UserFeedback object from a MovieLens dataset line. These are
		tab separated as:
		user id | item id | rating | timestamp.

		Inputs:
			tsl (str): Feedback line with tab-separated values.

		Returns:
			(UserFeedback): The constructed user feedback object.
		'''
		user_id, item_id, rating, timestamp = tsl.split("\t")

		user_id = int(user_id)
		item_id = int(item_id)
		rating = int(rating)
		timestamp = int(timestamp)

		return UserFeedback(user_id, item_id, rating, timestamp)


	def __init__(self, user_id=None, item_id=None, rating=None, \
		timestamp=None):
		'''
		Create the UserFeedback object, item_id and rating must be integers 
		(this is to work with the RBM models).

		Inputs:
			user_id (str|int): The user id.
			item_id: (int): The item id.
			rating (int): The rating.
			timestmap (int): The timestamp.
		'''
		self.user_id = user_id
		self.item_id = item_id
		self.rating = rating
		self.timestamp = timestamp
