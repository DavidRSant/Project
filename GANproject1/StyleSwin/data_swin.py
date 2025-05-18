import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import lmdb
import subprocess
import sys
import timm
from io import BytesIO

from torch.utils.tensorboard import SummaryWriter

# Variáveis de ambiente para compilação no Windows
os.environ["DISTUTILS_USE_SDK"] = "1"
os.environ["MSSdk"] = "1"
os.environ["VSINSTALLDIR"] = r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
os.environ["VCINSTALLDIR"] = os.path.join(os.environ["VSINSTALLDIR"], "VC")
os.environ["VCToolsVersion"] = "14.4"  # Verifique em "%VCINSTALLDIR%\Tools\MSVC\"
os.environ["PATH"] = f"{os.environ['VCINSTALLDIR']}\\Tools\\MSVC\\{os.environ['VCToolsVersion']}\\bin\\Hostx64\\x64;{os.environ['PATH']}"

# Config
NPZ_PATH = r"C:\Users\David\Downloads\celeba_faces_50k_subset.npz"
TEMP_IMG_DIR = "celeba_images_tmp"
LMDB_PATH = "celeba_faces_128x128.lmdb"
IMAGE_SIZE = 128
BATCH_SIZE = 4
N_SAMPLE = 25
OUTPUT_PATH = "models_output"
STYLESWIN_DIR = r"C:\Users\David\Downloads\StyleSwin-main"
PYTHON_EXECUTABLE = sys.executable  # usa o mesmo Python do ambiente atual



# Step 1: Extract images from .npz to .png
def extract_npz_to_png(npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    key = 'faces' if 'faces' in data else 'images'
    images = data[key]

    for idx, img_arr in enumerate(tqdm(images, desc="Salvando PNGs")):
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)

        img = Image.fromarray(img_arr)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        img.save(os.path.join(output_dir, f"{idx:05}.png"), format="PNG")

def convert_images_to_lmdb(image_dir, lmdb_path, resolution):
    import shutil
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)  # limpa LMDB anterior corrompido

    env = lmdb.open(lmdb_path, map_size=int(10e9))
    txn = env.begin(write=True)
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    valid_count = 0

    for idx, fname in enumerate(tqdm(img_files, desc="Criando LMDB")):
        path = os.path.join(image_dir, fname)
        try:
            with open(path, 'rb') as f:
                raw = f.read()
                # Verifica se a imagem é válida antes de salvar
                Image.open(BytesIO(raw)).verify()

                key = f"{resolution}-{str(valid_count).zfill(5)}".encode("utf-8")

                txn.put(key, raw)
                valid_count += 1

        except Exception as e:
            print(f"[IGNORADA] Imagem inválida {fname}: {e}")

        if (valid_count + 1) % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b'length', str(valid_count).encode('utf-8'))
    txn.put(b'resolution', str(resolution).encode('utf-8'))
    txn.commit()
    env.close()

    print(f"✔️ LMDB criado com {valid_count} imagens válidas.")



def sanity_check_lmdb(lmdb_path, resolution):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        key = f"{resolution}-00000".encode("utf-8")
        raw = txn.get(key)
        if raw is None:
            raise ValueError(f"Imagem com chave '{key.decode()}' não encontrada no LMDB!")
        try:
            img = Image.open(BytesIO(raw))
            img.verify()
        except Exception as e:
            raise RuntimeError(f"Erro ao decodificar imagem '{key.decode()}': {e}")
    print("✔️ Sanidade LMDB verificada com sucesso.")



if __name__ == "__main__":
    #extract_npz_to_png(NPZ_PATH, TEMP_IMG_DIR)
    convert_images_to_lmdb(TEMP_IMG_DIR, LMDB_PATH, IMAGE_SIZE)
    sanity_check_lmdb(LMDB_PATH, IMAGE_SIZE)


  