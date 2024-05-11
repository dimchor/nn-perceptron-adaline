import torch

import perceptron
import dataset
import stats
import constants

def main():
    # stats.demo_generate_random_values()

    # stats.demo_cities()

    stats.demo_hospital()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

if __name__ == "__main__":
    main()
