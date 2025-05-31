
import torch

def softmax(x, dim=None, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    return torch.softmax(x, dim=dim)


if __name__ == "__main__":
  # generate a random vector of values
  x = torch.randn(10)
  soft_x = softmax(x, dim=0)
  print(x, soft_x, soft_x.sum())
  # generate a matrix 3x10 of random values
  print("===============")
  mx = torch.randn(3, 5, 4)
  soft_mx = softmax(mx, dim=1)
  print(mx, soft_mx) 
  print(soft_mx.sum(dim=0), soft_mx.sum(dim=1), soft_mx.sum(dim=2))

