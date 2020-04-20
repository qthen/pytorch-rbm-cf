import torch
import torch.nn.functional as F
from models.rbm_cf import RbmCf
from utils.utils import (
	one_hot_encode, 
	create_visible_binary_indicator_matrix, 
	check_visible_user_ratings_ok
)


class BernoulliRbmCf(RbmCf):

	'''
	RBM with visible softmax units and hidden Bernoulli units, an 
	implementation of the same model as introduced by the paper.

	We reconstruct the entire input (which costs more space) but allows us to 
	do fast matrix multiplication.

	Inputs:
		M (int): The number of items.
		K (int): The number of rating categories.
		F (int): The number of hidden units.
	'''
	def __init__(self, M, K, F):
		super().__init__(M, K, F)

		# Hmmm. I described W as a 2d tensor..? umm....
		# W describes the weights from all the visible units to the hidden
		# units.
		self._W = torch.zeros(self._M, self._K, self._F)
		self._W.normal_(mean=0, std=0.01)

		# Biases on hidden units.
		self._B_h = torch.zeros(self._F,)

		# Biases on the visible softmax units.
		# Each B_v[i,j] is the bias for rating k on movie i.
		self._B_v = torch.zeros(self._M, self._K)

		# For momentum.
		self._W_p = torch.zeros(self._W.shape)
		self._B_h_p = torch.zeros(self._B_h.shape)
		self._B_v_p = torch.zeros(self._B_v.shape)


	def p_v_given_h(self, H):
		'''
		Returns the probabilites of the visible softmax units given the hidden 
		units of the movies we want to reconstruct.

		B - Batch size
		M - Item size
		F - Hidden unit size

		Inputs:
			H (tensor): B x F tensor of hidden units.
		Returns:
			(tensor): B x M x K tensor normalized probabilities. 
		'''

		# Result: WH, W is (MK)xF, H is FxB => MKxB
		# Tranpose to get row vectors (BxMK), then seperate into visible units.
		V = (torch.mm(self._W.view(self._M*self._K, self._F), H.T) + \
			self._B_v.view(-1, 1)).T.view(-1, self._M, self._K)

		# Normalize probabilities.
		V = F.softmax(V, dim=2)

		return V


	def p_h_given_v(self, V):
		'''
		Return the probabilities of the hidden units given the visible units:
		p(h_j = 1|V).

		Note when a rating is missing for an item, its row vector is all zero.
		This allows missing ratings to have no effect on the hidden units.

		B - batch size
		M - item size
		K - rating size
		F - hidden unit size

		Inputs:
			V (tensor): B x M x K - User ratings where a row is all zero if no
			 ratings were observed.

		Returns:
			(tensor): B x F - Hidden unit probabilities (B of them).
		'''

		# Result: W^TV, W is (MK)xF, V is (MK)xB => FxB.
		# Transpose the result to return BxF: row vectors are hidden units.
		return torch.sigmoid(torch.mm(self._W.view(-1, self._F).T, V.view(-1,
			self._M*self._K).T) + self._B_h.view(-1, 1)).T


	def sample_h_given_v(self, V):
		'''
		Samples a hidden vector given visible units. 

		Inputs:
			V (tensor): B x M x K - User ratings where a row is all zero if no
			 ratings were observed. 

		Returns:
			(tensor): B x H tensor of sampled hidden units.
		'''
		H = self.p_h_given_v(V)
		return H.bernoulli()


	def sample_v_given_h(self, H):
		'''
		Samples the visible units given the hidden units.

		Inputs:
			H (tensor): B x F - Batch of hidden units.

		Returns:
			(tensor): B x M x K - Sampled visible units with each item
			describing the category (rating) that was sampled.
		'''
		
		# V is B x M x K of normalized probabilities.
		V = self.p_v_given_h(H)

		# Collapse V to make it easier to sample, then sample.
		sampled_ratings = V.view(-1, self._K).multinomial(num_samples=1).view(
			-1,)

		# One hot encode the and reshape into the visible softmax units.
		# TODO: It might be better to return this NOT one hot encoded because
		# then we don't have to copy this again when we mask it, we can just 
		# one hot encode the ones we care about.
		return one_hot_encode(sampled_ratings, \
			num_categories=self._K).view(-1, self._M, self._K)


	def fit(self, visible_user_ratings_batch, T=0, lr=1e-3, momentum=0.9):
		'''
		Fits the RBM to a batch of user ratings using CD_T.

		Inputs:
			visible_user_ratings_batch (VisibleUserRatings): The visible user
			 ratings to train on (2D tensor batched).
			t (int): Number of iterations to run CD for.
			lr (float): The learning rate.
			momentum (int): Momentum for update.
		'''
		check_visible_user_ratings_ok(visible_user_ratings_batch)

		batch_size = len(visible_user_ratings_batch.ratings)

		# 3D indexing for faster masking.
		batch_indices = [batch_idx for batch_idx in range(len
			(visible_user_ratings_batch.item_indices)) for _ in range(len
			(visible_user_ratings_batch.item_indices[batch_idx]))]
		item_indices = [item_index for indices in
			visible_user_ratings_batch.item_indices for item_index in indices]
		rating_indices = [rating_index for indices in
			visible_user_ratings_batch.ratings for rating_index in indices]

		# One hot encode.
		V = torch.zeros(batch_size, self._M, self._K)
		V[batch_indices, item_indices, rating_indices] = 1.0

		# Probabilities of hidden units after single pass.
		H_t0_prob = self.p_h_given_v(V) # Probably don't need this.
		H_t0_sampled = H_t0_prob.bernoulli()

		# Compute positive gradient (<vh>_data) using batched outer product.
		# Use samples for hidden units (but probabilities could be used).
		positive_gradient = torch.bmm(V.view(-1, self._M*self._K).unsqueeze(2),
			H_t0_sampled.unsqueeze(1)).view(-1, self._M, self._K,
				self._F)

		# Do Gibbs sampling for T steps to get an approximation for <vh>_model.
		# Use samples for hidden units for final step (but probabilties could
		# be used).
		H_tN_prob = H_t0_prob
		H_tN_sampled = H_t0_sampled
		V_tN_prob = None
		V_tN_sampled = None
		for _ in range(T):
			# Backward pass.
			V_tN_prob = self.p_v_given_h(H_tN_sampled)

			# Sample visible units with missing items masked.
			sampled_rating_indices = V_tN_prob[batch_indices,
				item_indices].multinomial(num_samples=1).view(-1)

			# Reallocating a new tensor is the same as re-using the tensor...
			V_tN_sampled = torch.zeros(batch_size, self._M, self._K)
			V_tN_sampled[batch_indices, item_indices, \
				sampled_rating_indices] = 1.0

			# Forward pass.
			H_tN_prob = self.p_h_given_v(V_tN_sampled)
			H_tN_sampled = H_tN_prob.bernoulli()

			# When sampling we need to use the activation vector itself since 
			# if we use the probabilities, hidden units can pass real-valued
			# information to the visible units which violates the bottleneck
			# principle.

		# Compute <vh>_model using batched outer product.
		# Use samples for visible and hidden units (but probabilites could be
		# used) - however they tend to lead to worse density models.
		# https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
		negative_gradient = torch.bmm(V_tN_sampled.view(-1,
			self._M*self._K).unsqueeze(2), H_tN_sampled.unsqueeze(1)).view(-1, self._M, self._K, self._F)

		# Update to W.
		d_W = (positive_gradient - negative_gradient).sum(dim=0)

		# Gradient of visible unit biases is approximated by:
		# grad(b_i^k) := v_j - v'_j^(n): The given visible unit subtracted by
		#  its probability after n passes.
		d_B_v = torch.zeros(self._B_v.shape)
		# We don't need V anymore so might as well re-use it here for visible
 		# unit biases updates.
 		# V is already masked and this subtraction only considers the relevant
 		# item indices for each example.
		V[batch_indices, item_indices] -= V_tN_prob[batch_indices,item_indices]
		d_B_v = V.sum(dim=0)

		# Gradient of hidden unit biases is approximated by:
		# grad(c_j) := h_j^0 - h_j^n: The hidden unit probabilities after the
		#  first pass subtracted by the hdden unit probabilities after n 
		#  passes.
		d_B_h = (H_t0_prob - H_tN_prob).sum(dim=0)

		# Divide all gradients by the frequency of the items in the batch (this
		# is basically the number of times they "would" have been updated).
		# Dividing them by the batch_size underestimates gradients for items
		# that are rated rarely, but this shouldn't matter too much since
		# learning rate differs by multiple magnitudes while batch_size is
		# typically small in comparison - however this gets us the most out of
		# the gradient updates (but requires extra time to compute the item
		# frequency).
		#
		# TODO: Implement weight decay.
		
		# Use ones to prevent division by zero. Since updates are already
		# masked for missing items, this is fine.
		item_freq = torch.zeros(batch_size, self._M)
		item_freq[batch_indices, item_indices] = 1
		item_freq = item_freq.sum(dim = 0)
		item_freq[item_freq == 0] = 1

		# Momentum updates.
		self._W_p = momentum*self._W_p + lr*(d_W/item_freq.view(-1, 1, 1))
		self._B_h_p = momentum*self._B_h_p + lr*(d_B_h/batch_size)
		self._B_v_p = momentum*self._B_v_p + lr*(d_B_v/item_freq.view(-1, 1))

		self._W += self._W_p
		self._B_v += self._B_v_p
		self._B_h += self._B_h_p



	def _mask_visible_units(self, V_batch, item_indices_batch):
		'''
		Masks each visible matrix row in V_batch where the item is not a
		respective item_indices.

		This is to discard ratings for items that are considered "missing".

		Inputs:
			V_batch (tensor): B x M x K tensor.
			item_indices_batch ([[Int]]): B-lists of list of indices.

		Returns:
			B x M x K tensor
		'''

		if V_batch.shape[0] != len(item_indices_batch):
			raise ValueError("item_indices_batch length must equal number " \
				"of visible matrices batch (V_batch).")

		V_batch_masked = torch.zeros(V_batch.shape)

		for i in range(V_batch.shape[0]):
			# Only add the relevant items per user in batch.
			V_batch_masked[i][item_indices_batch[i]] = V_batch[i][item_indices_batch[i]]

		return V_batch_masked



