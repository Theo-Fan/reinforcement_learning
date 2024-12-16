import ray

ds = ray.data.read_text("data/test.txt")

print(ds.schema())



print("\n##############################################")
ds = ray.data.from_items([
    {"food": "spam", "price": 9.34},
    {"food": "ham", "price": 5.37},
    {"food": "eggs", "price": 0.94}
])

print(ds)



print("\n##############################################")
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

tds = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
ds = ray.data.from_torch(tds)

print(ds)