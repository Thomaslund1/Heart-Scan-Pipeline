import os
import json
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion, median_filter
from skimage.morphology import erosion, dilation, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional downsample function
def downsample(arr):
    # 1:2 downsample 
    n2 = arr.shape[0] // 2
    m2 = arr.shape[1] // 2
    arr = arr[:2*n2, :2*m2]
    blocks = arr.reshape(n2, 2, m2, 2).transpose(0, 2, 1, 3)
    return np.median(blocks, axis=(2, 3))

binary_data = True 

def load_tiff_stack(folder, binary_data=False):
    # Get sorted list of TIFF files and trim the edges
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))])[10:-10]
    assert files, f"No TIFF files found in {folder}"
    first = tiff.imread(os.path.join(folder, files[0]))
    h, w = downsample(first).shape
    d = len(files)
    print(f"Reading stack {w}×{h}×{d}")
    volume = np.zeros((d, h, w), dtype=np.uint8)

    # ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        # Start reading and downsampling each image in parallel
        for i, fname in enumerate(files):
            futures.append(executor.submit(process_image, os.path.join(folder, fname), i, binary_data))
        
        # Collect results as they are completed
        for future in tqdm(as_completed(futures), total=len(futures)):
            img, i = future.result()
            volume[i] = img

    return volume

def process_image(file_path, index, binary_data):
    # Read the image
    img = tiff.imread(file_path)
    if binary_data:
        img = (img > 0).astype(np.uint8) * 255  # Binary mask conversion
    # Downsample the image
    downsampled_img = downsample(img)
    return downsampled_img, index

def save_raw(volume, out_prefix):
    raw_path = out_prefix + ".raw"
    meta_path = out_prefix + ".json"
    volume.tofile(raw_path)
    meta = {"width": volume.shape[2],
            "height": volume.shape[1],
            "depth": volume.shape[0],
            "format": "R8"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved RAW data to {raw_path}")
    print(f"Saved metadata to {meta_path}")

def morph(volume, iterations):
    volume = volume.astype(bool)
    
    struct1 = np.zeros((5,5,5), bool)
    struct1[:, 3, 3] = 1  # z-axis line

    struct2 = np.zeros((5,5,5), bool)
    struct2[:, 2:4, 2:4] = 1  # z-axis square

    # First phase
    for _ in range(iterations):
        volume = binary_dilation(volume, footprint=struct1)
        volume = binary_erosion(volume, footprint=struct1)

    # Second phase
    for _ in range(5):
        volume = binary_dilation(volume, footprint=struct2)
        volume = binary_erosion(volume, footprint=struct2)

    return volume

def distBridge(arr, threshold=10):
    vol = arr.astype(bool)
    z_weight = 0.2
    y_weight = 1
    x_weight = 1
    dist = distance_transform_edt(~vol, sampling=(z_weight, y_weight, x_weight))

    # Fill any gap smaller than threshold
    connected = dist < threshold

    # Preserve original foreground
    return connected | vol

def apply_watershed(volume):
    print("Applying watershed segmentation...")
    labels = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        # Compute distance transform
        distance = distance_transform_edt(volume[i] > 0)
        # Find local maxima
        local_max = peak_local_max(distance, footprint=np.ones((3, 3)), labels=volume[i] > 0)
        # Watershed
        labels[i] = watershed(-distance, markers=local_max, mask=volume[i] > 0)
    return labels

def process_volume(volume, operation='morphological'):
    if operation == 'morphological':
        volume = morph(volume, 5)
    elif operation == 'adt':
        volume = distBridge(volume)
    else:
        print(f"Unknown operation {operation}")
    return volume

def main():
    processing_method = 'morphological'
    if len(sys.argv) < 3:
        print('Usage: python tiffToRaw.py {inputFolder} {outputFolder} {processingMethod="morphological"}')
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if len(sys.argv) > 3:
        processing_method = sys.argv[3]
    
    if processing_method not in ['morphological', 'adt']:
        print(f"Invalid processing method: {processing_method}. Choose 'morphological' or 'adt'.")
        sys.exit(1)
    
    # Process the TIFF stack
    vol = load_tiff_stack(input_folder)
    vol_processed = np.array(process_volume(vol, processing_method), dtype=np.bool)
    
    # Save the processed volume
    save_raw(vol_processed, output_folder)

if __name__ == "__main__":
    main()

