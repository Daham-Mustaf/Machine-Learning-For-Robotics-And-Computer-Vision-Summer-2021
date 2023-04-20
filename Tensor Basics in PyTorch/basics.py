import torch

# Create tensor A with shape (3, 4) and random values drawn from a uniform distribution between 0 and 1
A = torch.rand((3, 4))

# Create tensor B with shape (3, 4) and random values drawn from a uniform distribution between 0 and 1
B = torch.rand((3, 4))

C = A + B
D = A * B
# Compute the dot product of A and B
F = torch.mm(A, B.t())

# Reshape F to have shape (4, 3)
G = F.reshape(4, 3)

# Compute the element-wise exponential of tensor G
H = torch.exp(G)

# Compute the sum of tensor H along the second dimension
I = torch.sum(H, dim=1)

