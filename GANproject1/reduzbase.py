import numpy as np
from tqdm import tqdm

def load_and_save_subset(input_path, output_path, num_images=50000):
    """
    Carrega um subconjunto de imagens do arquivo NPZ e salva em um novo arquivo.
    
    Args:
        input_path (str): Caminho para o arquivo NPZ de entrada
        output_path (str): Caminho para o arquivo NPZ de saída
        num_images (int): Número de imagens para extrair (padrão: 50000)
    """
    # Carregar o arquivo original
    print(f"Carregando dados de {input_path}...")
    with np.load(input_path) as data:
        # Verificar quantas imagens existem no arquivo
        total_images = data['faces'].shape[0]
        print(f"Total de imagens no arquivo original: {total_images}")
        
        # Verificar se o número solicitado é válido
        if num_images > total_images:
            raise ValueError(f"O arquivo contém apenas {total_images} imagens, mas {num_images} foram solicitadas.")
        
        # Carregar os metadados (se existirem)
        metadata = {}
        for key in data.files:
            if key != 'faces':
                metadata[key] = data[key]
        
        # Carregar as imagens em lotes para economizar memória
        print(f"Extraindo {num_images} imagens...")
        images = data['faces'][:num_images]
        
        # Se houver outros arrays no arquivo NPZ, mantenha-os (mas apenas para as imagens selecionadas)
        additional_data = {}
        for key in metadata:
            if metadata[key].shape[0] == total_images:
                additional_data[key] = metadata[key][:num_images]
            else:
                additional_data[key] = metadata[key]
        
    # Salvar o subconjunto em um novo arquivo NPZ
    print(f"Salvando {num_images} imagens em {output_path}...")
    np.savez_compressed(output_path, images=images, **additional_data)
    
    print("Processo concluído com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    input_file = "celeba_faces_rtx3060_optimized.npz"
    output_file = "celeba_faces_50k_subset.npz"
    
    load_and_save_subset(input_file, output_file, num_images=50000)