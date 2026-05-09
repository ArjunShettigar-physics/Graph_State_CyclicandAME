import numpy as np

# Load the .npy file
input_path = "ame_18_5_okayfinal.npy"
output_path = "ame_18_5_okayfinal.txt"

data = np.load(input_path, allow_pickle=True)


n_matrices = data.shape[0]
print(n_matrices)
with open(output_path, "w") as f:
    for matrix in (data):
        for row in matrix:
            f.write(" ".join(str(val) for val in row) + "\n")
        # Add 2 blank lines after every matrix (including the last)
        f.write("\n\n")

print(f"Done! Converted {n_matrices} matrix/matrices → '{output_path}'")
