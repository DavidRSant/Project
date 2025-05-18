import os
import sys
import subprocess
from pathlib import Path
import argparse

# ===== Configuração de variáveis de ambiente =====
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
os.environ["PATH"] += ";" + os.path.join(os.environ["CUDA_PATH"], "bin")

os.environ["VSINSTALLDIR"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
os.environ["VCINSTALLDIR"] = os.path.join(os.environ["VSINSTALLDIR"], "VC")

msvc_dir = Path(os.environ["VCINSTALLDIR"]) / "Tools" / "MSVC"
versions = [d.name for d in msvc_dir.iterdir() if d.is_dir()]
if not versions:
    raise FileNotFoundError(f"Nenhuma versão encontrada em {msvc_dir}")
versions.sort(reverse=True)
os.environ["VCToolsVersion"] = versions[0]

compiler_path = msvc_dir / os.environ["VCToolsVersion"] / "bin" / "Hostx64" / "x64"
os.environ["PATH"] += f";{compiler_path}"

os.environ["PATH"] += ";" + r"C:\Users\David\anaconda3\envs\gpu\Scripts"

# ===== Configuração do projeto =====
NPZ_PATH = r"C:\Users\David\Downloads\celeba_faces_50k_subset.npz"
TEMP_IMG_DIR = "celeba_images_tmp"
LMDB_PATH = "celeba_faces_128x128.lmdb"
IMAGE_SIZE = 128
BATCH_SIZE = 4
N_SAMPLE = 25
OUTPUT_PATH = "models_output"
STYLESWIN_DIR = r"C:\Users\David\Downloads\StyleSwin-main"
PYTHON_EXECUTABLE = sys.executable

def train_styleswin(lmdb_path, max_iter, eval_freq, save_freq):
    import os
    import time
    import subprocess
    import sys

    # Diretórios
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    checkpoint_dir = os.path.join(OUTPUT_PATH, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_script = os.path.join(STYLESWIN_DIR, "train_styleswin.py")

    env = os.environ.copy()
    env["WORLD_SIZE"] = "1"

    print("\U0001F527 Executando treinamento com StyleSwin...")
    print(f"\U0001F4BE Salvando checkpoints a cada {save_freq} iterações em {checkpoint_dir}.")
    print(f"\U0001F4CA Avaliação a cada {eval_freq} iterações.")
    print(f"\U0001F501 Máximo de iterações: {max_iter}")

    subprocess.run([
        sys.executable, train_script,
        "--path", lmdb_path,
        "--lmdb",
        "--size", str(IMAGE_SIZE),
        "--batch", str(BATCH_SIZE),
        "--sample_path", OUTPUT_PATH,
        "--eval_gt_path", r"C:\\Users\\David\\Downloads\\StyleSwin-main\\celeba_images_tmp",
        "--tf_log",
        "--workers", "0",
        "--iter", str(max_iter),
        "--eval_freq", str(eval_freq),
        "--save_freq", str(save_freq),
        "--checkpoint_path", checkpoint_dir  # <- checkpoint_dir definido acima
    ], cwd=STYLESWIN_DIR, env=env, check=True)

    print("\u2705 Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=80000, help='Número máximo de iterações')
    parser.add_argument('--eval_freq', type=int, default=10000, help='Frequência de avaliação (em iterações)')
    parser.add_argument('--save_freq', type=int, default=10000, help='Frequência de salvamento de checkpoints')

    args = parser.parse_args()

    print("🔍 Configuração do ambiente finalizada.")
    print(f"✅ MSVC versão detectada: {os.environ['VCToolsVersion']}")
    print(f"✅ Compiler path adicionado ao PATH: {compiler_path}")
    print(f"✅ CUDA PATH: {os.environ['CUDA_PATH']}")
    print(f"✅ TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
    print("\n🚀 Iniciando treinamento...\n")

    train_styleswin(LMDB_PATH, args.max_iter, args.eval_freq, args.save_freq)
