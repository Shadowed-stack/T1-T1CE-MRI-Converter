import kagglehub
import shutil
import os

path = kagglehub.dataset_download(
    "awsaf49/brats2020-training-data"
)

print("Downloaded:", path)

dst = "dataset"

if os.path.exists(dst):
    shutil.rmtree(dst)

shutil.copytree(path, dst)

print(f"Dataset copied to ./{dst}")
