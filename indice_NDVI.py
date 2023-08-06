# Importe os módulos necessários.
import os
import rasterio  # Adicione esta linha para importar o módulo rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import dask.array as da
import dask

# Esta função lê as bandas NIR e RED de um arquivo GeoTIFF.
def read_bands(file_path, band_numbers):
    with rasterio.open(file_path) as src:
        # Leia as bandas especificadas.
        bands = [src.read(band_num) for band_num in band_numbers]
        # Obtenha o sistema de coordenadas de referência (CRS) e a transformação do arquivo.
        crs = src.crs
        transform = src.transform
    return bands, crs, transform

# Esta função calcula o índice NDVI a partir das bandas NIR e RED.
def calculate_ndvi(nir, red):
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = da.nan_to_num((nir - red) / (nir + red))
    return ndvi

# Esta função cria uma paleta de cores personalizada para o índice NDVI.
def generate_custom_cmap():
    # Altere as cores usadas na paleta para melhorar a aparência do mapa.
    colors = [(0.5, 0.0, 0.0),    # Vermelho mais escuro
              (0.9, 0.8, 0.0),    # Amarelo mais escuro
              (0.0, 0.4, 0.0)]    # Verde mais escuro
    return LinearSegmentedColormap.from_list("custom_ndvi_cmap", colors, N=256)

# Esta função plota o índice NDVI como uma imagem e o salva em um arquivo.
def plot_ndvi(ndvi_array, file_name, output_folder, crs, transform):
    # Defina os limites mínimos e máximos dos valores do índice NDVI.
    vmin, vmax = 0, 1
    # Crie uma paleta de cores personalizada.
    custom_cmap = generate_custom_cmap()
    custom_cmap.set_under(color='white', alpha=None)
# Crie uma figura Matplotlib.
    plt.figure(figsize=(12, 8))
    # Obtenha os limites da imagem.
    extent = rasterio.transform.array_bounds(ndvi_array.shape[0], ndvi_array.shape[1], transform)
    # Defina a extensão da figura.
    map_extent = [extent[0] + transform[0] * 0.5, extent[1] - transform[0] * 0.5,
                  extent[2] + transform[4] * 0.5, extent[3] - transform[4] * 0.5]
# Plote a imagem do índice NDVI.
    im = plt.imshow(ndvi_array, cmap=custom_cmap, vmin=vmin, vmax=vmax, extent=map_extent, aspect='auto')
# Crie uma barra de cores para a imagem.
    cbar = plt.colorbar(im, ticks=np.linspace(0, 1, 5))
    # Defina o rótulo da barra de cores.
    cbar.set_label('NDVI', rotation=270, labelpad=20)
# Defina o título da figura.
    plt.title(f'Mapa NDVI - {file_name}')
    # Defina os rótulos dos eixos X e Y.
    plt.xlabel('Coluna')
    plt.ylabel('Linha')
# Salve a imagem em um arquivo.
    output_file_tiff = os.path.join(output_folder, f"NDVI_{os.path.splitext(file_name)[0]}.tif")
    with rasterio.open(output_file_tiff, 'w', driver='GTiff', height=ndvi_array.shape[0], width=ndvi_array.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform) as dst:
        dst.write(ndvi_array, 1)

    output_file_png = os.path.join(output_folder, f"NDVI_{os.path.splitext(file_name)[0]}.png")
    plt.savefig(output_file_png, dpi=600, bbox_inches='tight')
    plt.close()

def process_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_folder, file_name)
            bands, crs, transform = read_bands(input_path, [4, 3])
            nir, red = bands

            nir_dask = da.from_array(nir, chunks=(512, 512))
            red_dask = da.from_array(red, chunks=(512, 512))

            ndvi_array = calculate_ndvi(nir_dask, red_dask).compute()

            ndvi_array[ndvi_array == 0] = np.nan

            os.makedirs(output_folder, exist_ok=True)

            plot_ndvi(ndvi_array, file_name, output_folder, crs, transform)

def main():
    input_folder = input("Digite o caminho da pasta com os arquivos de entrada: ")
    output_folder = input("Digite o caminho da pasta de saída dos arquivos NDVI: ")

    # Configuração para otimização Dask (processamento paralelo)
    # Configuração para otimização Dask (processamento paralelo)
    dask.config.set(scheduler='threads')  # Utilize threads para paralelizar as operações Dask

    process_files(input_folder, output_folder)

if __name__ == "__main__":
    main()
    
