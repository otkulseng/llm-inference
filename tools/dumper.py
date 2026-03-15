# Your logic for dumping the weights goes here. 
# You have to prepare binary files, in a format that you use for loading them in the C++ files. 
# You may use any library listed in requirements.txt
# Read the project description for more information.


from safetensors.torch import load_file
import torch
import struct
from pathlib import Path
import numpy as np
from tqdm import tqdm

F16 = 0
F32 = 1
BF16 = 2


def write_tensor(path: str | Path, t: torch.Tensor) -> None:   
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    t = t.detach().cpu().contiguous()
    assert t.dim() == 2

    rows, cols = t.shape

    with open(path, "wb") as f:
        if t.dtype == torch.float16:
            dtype_code = F16
            arr = t.numpy()
        elif t.dtype == torch.float32:
            dtype_code = F32
            arr = t.numpy() 
        elif t.dtype == torch.bfloat16:
            # Seeing as numpy does not support BF16, I view it as an int16
            # and store that. While it becomes the wrong number, the underlyig
            # bytes are the same and we can therefore load correctly on the cpp
            # side.
            dtype_code = BF16
            arr = t.view(torch.int16).numpy()
        else:
            raise TypeError(f"unsupported dtype: {t.dtype}")

        nbytes = arr.nbytes

        f.write(struct.pack("<IQQQ", dtype_code, rows, cols, nbytes))

        # raw bytes
        arr.tofile(f)  # binary when sep=""

root = Path("./assets/llama3")
store = root / "blobs"
store.mkdir(exist_ok=True)

for file in root.iterdir():
    if not file.name.endswith(".safetensors"):
        continue
    
    loaded = load_file(file)
    
    for name, tensor in tqdm(loaded.items()):
        if not isinstance(tensor, torch.Tensor):
            print(f"{name} not a tensor!")
            continue
        write_tensor(store / name, tensor)

