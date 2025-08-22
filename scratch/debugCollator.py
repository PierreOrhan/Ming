#%%
from datasets import IterableDataset
class Mygen:
    def __call__(self):
        for _ in range(10):
            yield 0

def gen():
    for _ in range(10):
        yield 0

l = IterableDataset.from_generator(Mygen)

# %%
