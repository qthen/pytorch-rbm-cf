import time
import torch
import random

# Is reusing the same tensor faster or creaitng a new one for each loop?
# Zero out in each loop average time: 0.003
# Re-allocate new tensor for each loop after time: 0.003
# LOL...


BATCH_SIZE = 32
ITEM_SIZE = 1000
RATING_SIZE = 5
T = 10
TRIES = 1000

item_indices = [random.sample(range(ITEM_SIZE), k=random.randint(10, 100)) for _ in range(BATCH_SIZE)]
rating_indices = []
for indices in item_indices:
	ratings = [random.randint(0,4) for _ in range(len(indices))]
	rating_indices.append(ratings)
batch_indices = [batch_idx for batch_idx in range(len(item_indices)) for _ in range(len(item_indices[batch_idx]))]

# Flatten item_indices
flatten_item_indices = [item_idx for indices in item_indices for item_idx in indices]
flatten_rating_indices = [rating_idx for indices in rating_indices for rating_idx in indices]

# Zero out for each loop.
start_time = time.time()
for _ in range(TRIES):
	V = torch.zeros(BATCH_SIZE, ITEM_SIZE, RATING_SIZE)
	for _ in range(T):
		# Clear previous results.
		V[:,:,:] = 0
		V[batch_indices, flatten_item_indices, flatten_rating_indices] = 1.0
end_time = time.time()
print("Zero out in each loop average time: {:.3f}".format((end_time - start_time)/TRIES))

# Re-allocate new tensor after each loop.
start_time = time.time()
for _ in range(TRIES):
	V = torch.zeros(BATCH_SIZE, ITEM_SIZE, RATING_SIZE)
	for _ in range(T):
		V = torch.zeros(BATCH_SIZE, ITEM_SIZE, RATING_SIZE)
		V[batch_indices, flatten_item_indices, flatten_rating_indices] = 1.0
end_time = time.time()
print("Re-allocate new tensor for each loop after time: {:.3f}".format((end_time-start_time)/TRIES))

