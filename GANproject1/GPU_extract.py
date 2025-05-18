import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch
import logging
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

# Configurações otimizadas para GPU
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.ERROR)

def setup_gpu():
    """Configurações para RTX 3060"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.set_num_threads(1)
        torch.cuda.empty_cache()
        return device
    return torch.device('cpu')

def load_and_detect(args):
    filename, input_dir, target_size, detector = args
    try:
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        
        # Detecta faces - adicionando verificação robusta
        with torch.no_grad():
            boxes, probs = detector.detect(img)
        
        if boxes is None or len(boxes) == 0:
            return None
            
        # Pega a face com maior probabilidade
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        
        # Verificação adicional de coordenadas
        if x1 >= x2 or y1 >= y2:
            return None
            
        # Corta e redimensiona a face
        face = img.crop((x1, y1, x2, y2))
        face = face.resize(target_size)
        
        return np.array(face).astype(np.float32) / 255.0
        
    except Exception as e:
        print(f"Erro ao processar {filename}: {str(e)}")
        return None

def process_rtx3060_optimized(input_dir, output_path, target_size=(128, 128), workers=8):
    device = setup_gpu()
    print(f"🚀 Usando dispositivo: {device}")
    
    # Configuração do MTCNN com tratamento de erro
    try:
        detector = MTCNN(
            device=device,
            keep_all=False,
            post_process=False,
            select_largest=False,
            #thresholds=[0.6, 0.7, 0.7]
        ).eval()
    except Exception as e:
        print(f"Erro ao inicializar MTCNN: {str(e)}")
        return

    # Lista de arquivos de imagem com verificação
    try:
        image_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except Exception as e:
        print(f"Erro ao ler diretório: {str(e)}")
        return

    if not image_files:
        print("Nenhuma imagem encontrada no diretório!")
        return

    print(f"🔍 Encontradas {len(image_files)} imagens para processar")
    
    faces = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        args = [(f, input_dir, target_size, detector) for f in image_files]
        
        with tqdm(total=len(image_files), 
                 desc="Processando Faces",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for result in executor.map(load_and_detect, args):
                if result is not None:
                    faces.append(result)
                pbar.update(1)
    
    if len(faces) > 0:
        try:
            faces_array = np.array(faces, dtype=np.float16)
            np.savez_compressed(output_path, faces=faces_array)
            print(f"\n✅ Processo concluído! Faces detectadas: {len(faces_array)}")
            print(f"💾 Arquivo salvo em: {output_path}")
        except Exception as e:
            print(f"\n❌ Erro ao salvar arquivo: {str(e)}")
    else:
        print("\n❌ Nenhuma face detectada nas imagens!")

if __name__ == '__main__':
    config = {
        'input_dir': r"C:\Users\David\Downloads\archive\img_align_celeba\img_align_celeba",
        'output_path': r"C:\Users\David\Downloads\celeba_faces_rtx3060_optimized.npz",
        'target_size': (128, 128),
        'workers': 8
    }
    process_rtx3060_optimized(**config)