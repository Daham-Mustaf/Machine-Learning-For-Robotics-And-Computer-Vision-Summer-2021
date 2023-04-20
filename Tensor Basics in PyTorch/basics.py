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


# Gradient Calculation With Autograd


# Create a tensor with requires_grad=True to track computation history and enable autograd
x = torch.tensor([3.0], requires_grad=True)

# Define a function to compute y
def compute_y(x):
    y = 2 * x ** 2 + 3 * x + 1
    return y

# Compute y and assign the result to y
y = compute_y(x)

# Compute the gradients of y with respect to x
y.backward()

# Print the gradients of x
print(x.grad)


