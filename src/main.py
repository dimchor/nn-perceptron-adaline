import torch
import perceptron
import dataset
import rng


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    rng.generate_random_values()

if __name__ == "__main__":
    main()
