import argparse
from safetensors import safe_open
from safetensors.torch import save_file


## Update a metadata key with a value. 

# !!!!
# NOTE 
# !!!!
# Overwrites the input file.

def main(args):
    with safe_open(args.safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()

        metadata[args.key] = args.value
        print(f"Updated {args.key} with {args.value}")

        tensors = {}

        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    save_file(tensors, args.safetensors_file, metadata)

    print(f"Saved to {args.safetensors_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("safetensors_file")
    argparser.add_argument("--key", help="Key to change in the metadata")
    argparser.add_argument("--value", help="Value to set to the metadata")

    args = argparser.parse_args()

    main(args)
