import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import perceptron
import dataset
import stats
import constants

def part_a():
    stats.demo_generate_random_values()

    stats.demo_cities()

    stats.demo_hospital()

def part_b():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


def main():

    # part_a()

    part_b()


if __name__ == "__main__":
    main()
