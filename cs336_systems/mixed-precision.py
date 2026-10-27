import torch

def toy_example():
    s = torch.tensor(0.0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0.0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0.0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0.0, dtype=torch.float16)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float16)
    print(s)


if __name__ == "__main__":
    with torch.device("mps"):
        toy_example()
        """results:
        tensor(10.0001, device='mps:0')
        tensor(9.9531, device='mps:0', dtype=torch.float16)
        tensor(9.9531, device='mps:0', dtype=torch.float16)
        tensor(9.9531, device='mps:0', dtype=torch.float16)
        """
    
