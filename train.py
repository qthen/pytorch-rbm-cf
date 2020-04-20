import torch
import time
from models.bernoulli_rbm_cf import BernoulliRbmCf
from utils.user_feedback_db import UserFeedbackDb
from utils.utils import one_hot_encode, create_visible_binary_indicator_matrix
import torch.nn as nn
from datasets.movielens_mini_batch_dataset import MovielensMiniBatchDataset


MOVIELENS_DATASET_FP = "data/ml-100k/u.data"
HIDDEN_UNIT_SIZE = 64
EPOCHS = 500
LR = 0.01
BATCH_SIZE = 32

def get_t(epoch):
	if epoch > 20:
		return 3
	if epoch > 30:
		return 5
	return 1


feedback_db = UserFeedbackDb.create_from_movielens(MOVIELENS_DATASET_FP)


model = BernoulliRbmCf(M = feedback_db.item_size() + 1, K = 5, F=HIDDEN_UNIT_SIZE)
loss = nn.MSELoss()

dataset = MovielensMiniBatchDataset(data_fp=MOVIELENS_DATASET_FP, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
	i = 0
	if epoch % 10 == 0:
		LR = max(LR/2.0, 1e-5)
	for batch in dataset.get_batch():
		start_time = time.time()
		model.fit(batch, T=get_t(epoch), lr=LR, momentum=0.9)
		end_time_fit = time.time()

		if i % 10 == 0:
			# Attempt a reconstruction to see what the loss looks like.
			V = create_visible_binary_indicator_matrix(batch, feedback_db.item_size()+1, 5)
			H = model.sample_h_given_v(V)
			reconstruction_probabilities = model.p_v_given_h(H)

			# Print an example.
			print("Rating is {}, predicted probability is {:.3f} in tensor ".format(batch.ratings[0][0], reconstruction_probabilities[0, batch.item_indices[0][0], batch.ratings[0][0]].item()), reconstruction_probabilities[0, batch.item_indices[0][0]])
			reconstruction = torch.log(reconstruction_probabilities)

			# Only consider the units that are relevant.
			reconstruction = model._mask_visible_units(reconstruction, batch.item_indices)
			end_time_reconstruction = time.time()

			# Rough cross entropy loss.
			cross_entropy = V*reconstruction
			cross_entropy = cross_entropy.sum().item()
			cross_entropy = -1*(cross_entropy/sum([len(item_indices) for item_indices in batch.item_indices]))

			print("Epcoh {}, Mean cross entropy loss roughly {:.3f}".format(epoch, cross_entropy))

			running_loss = 0.0
			denom = 0
			for i in range(len(batch.ratings)):
				running_loss += ((V[i][batch.item_indices[i], batch.ratings[i]] - reconstruction[i][batch.item_indices[i], batch.ratings[i]])**2).sum().item()
				denom += len(batch.item_indices[i])
			end_time_mse = time.time()
			print("Epoch {}, MSE is {:.5f}".format(epoch, running_loss/denom))
			print("Fit time: {:.3f}, Reconstruction time: {:.3f}, MSE time: {:3f}".format(end_time_fit - start_time, end_time_reconstruction-end_time_fit, end_time_mse-end_time_reconstruction))

			# Umm a very rough RMSE for ratings.
			rating_values = torch.tensor([1, 2, 3, 4, 5])
			predicted_ratings = (reconstruction_probabilities*rating_values).sum(dim=2)
			running_mse_loss = 0.0
			denom = 0
			for i in range(0, len(batch.ratings)):
				running_mse_loss += ((torch.tensor(batch.ratings[i]) - predicted_ratings[i][batch.item_indices[i]])**2).sum().item()
				denom += len(batch.ratings[i])
			print("MSE rating loss is {:.3f}".format(running_mse_loss/denom))

		i+= 1



# for e in range(EPOCHS):
# 	for user_id, feedback_arr in feedback_db.items():
# 		ratings = torch.LongTensor([feedback.rating for feedback in feedback_arr]) - 1
# 		item_indices = torch.LongTensor([feedback.item_id for feedback in feedback_arr])

# 		model.train(ratings=ratings, movie_indices=item_indices, T=get_t(e), lr=LR)

# 	# Get the reconstruction error, this is an average MSE loss.
# 	running_loss = 0
# 	number_of_examples = 0
# 	for user_id, feedback_arr in feedback_db.items():
# 		ratings = torch.LongTensor([feedback.rating for feedback in feedback_arr]) - 1
# 		item_indices = torch.LongTensor([feedback.item_id for feedback in feedback_arr])

# 		V_target = one_hot_encode(ratings, 5)



# 		running_loss += loss(V_target, probabilities).item()
# 		number_of_examples += 1

# 	print("Epoch {}. Average MSE loss: {:.5f}".format(e, running_loss/number_of_examples))
