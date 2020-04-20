from utils.user_feedback import UserFeedback


class UserFeedbackDb(object):
	
	'''
	Main interface to store user ratings on items (movies) - basically a fake (naive) in-memory database.

	Contains various factory methods to construct it from a variety of data
	sources.
	'''

	@staticmethod
	def create_from_movielens(movielens_data_fp):
		'''
		Constructs the UserFeedbackDb object from the movielens source given the data_fp. 

		Inputs:
			movielens_data_fp (str): Filepath to the movielens dataset.

		Returns:
			(UserFeedbackDb): The UserFeedbackDb object.
		'''

		feedback_db = UserFeedbackDb()

		with open(movielens_data_fp, 'r') as fp:
			for line in fp:
				feedback_db.insert(UserFeedback.create_from_movielens(line))

		return feedback_db


	def __init__(self):
		'''
		Create the UserFeedbackDb object. _db stores mappings of user ids to 
		array of feedback. 
		'''
		self._db = {}


	def insert(self, user_feedback):
		'''
		Adds a UserFeedback into the database, does not check for duplicates.

		Inputs:
			user_feedback (UserFeedback): The user feedback to add.
		'''
		if user_feedback.user_id not in self._db:
			self._db[user_feedback.user_id] = []

		self._db[user_feedback.user_id].append(user_feedback)


	def items(self):
		'''
		For doing key, value for loops over the database.
		'''
		return self._db.items()


	def item_size(self):
		'''
		Returns the max item_id, computes this on the fly.

		Returns:
			(int): Max item id.
		'''

		running_max = float("-inf")

		for user_id, feedback_arr in self.items():
			running_max = max(running_max, max([feedback.item_id for feedback \
				in feedback_arr]))

		return running_max


	def user_size(self):
		'''
		Returns the max user_id, computes this on the fly.

		Returns:
			(int): The max user id. 
		'''

		return max(self._db.keys())


	def __iter__(self):
		return iter(self._db)