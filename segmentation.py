import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import glob
import tifffile
import copy
import scipy.signal 
from concurrent.futures import ThreadPoolExecutor

def normalize(arr):
    #Returns a scaled array setting the blackpoint to the 1% low value
    out = robust_scale_10_90(arr)
    out -= int(np.percentile(out,1))
    out[np.where(out < 0)] = 0
    return out

def robust_scale_10_90(arr):
    p1 = np.nanpercentile(arr, .1)
    p99 = np.nanpercentile(arr, 99.9)
    out = (arr - p1) / (p99 - p1)
    out -= np.min(out)
    out[np.where(out>1)] = 1
    out = np.array(out*255,dtype=int)
    return out

def blackpoint(arr,val):
    arr = arr - val
    arr[np.where(arr < 0)] = 0
    return normalize(arr)

def interp(arr):
    x = np.arange(arr.shape[0])
    mask = np.where(~np.isnan(arr))[0]
    inrp = PchipInterpolator(x[mask],arr[mask])
    return inrp(x)

def outlier2Nan(arr):
    arr2 = copy.copy(arr)
    mean = np.nanmean(arr2)
    std = np.nanstd(arr2)
    mask = np.where(np.abs(arr2-mean) > 3.2*std)[0]
    arr2[mask] = np.nan
    return arr2

def nonFloat2Nan(arr):
    out = np.empty_like(arr)
    for i,val in enumerate(arr):
        try:
            out[i] = float(val)
        except ValueError:
            out[i]=np.nan
            print(f'Removed {val} @ {i}')    
    return out

def to1Bit(arr):
    med = np.nanmedian(arr)
    out = np.zeros_like(arr)
    out[np.where(arr > med)] = 1
    return np.array(out,dtype=bool)

def oneHot(arr):
    unq = np.unique(arr)
    out = np.zeros((len(arr),len(unq)))
    for i,val in enumerate(arr):
        out[i][np.where(unq == val)[0][0]] = 1
    return out

def asinh_transform(arr, scale=1.0, blackpoint=0.0):
    #Inverse sinh intensity curve
    transformed = np.arcsinh(arr / scale)
    transformed -= blackpoint
    return transformed

def differenceOfMedians(arr,smallWindowSize,largeWindowSize):
    #Returns difference of two differently sized medians, filtering noise and backround structures out
    arr = np.array(arr,dtype=np.float32)
    a = scipy.signal.medfilt(arr,smallWindowSize)
    b = scipy.signal.medfilt(arr,largeWindowSize)
    return np.abs(b-a)

def processFile(f, outputDir,small,large):
    #Accepts one file path, processes it, writes data to output path
    dater = tifffile.imread(f)
    
    arr = differenceOfMedians(dater, small, large)
    processed_arr = blackpoint(arr, 13000)
    finalArr = to1Bit(processed_arr)
    
    outputPath = os.path.join(outputDir, os.path.basename(f))
    tifffile.imwrite(outputPath, finalArr)
    print(f'Processed: {os.path.basename(f)}')

def processFiles(flist,outputDir,small = 3,large = 17):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(processFile, f, outputDir,small,large) for f in flist]
    for future in futures:
        future.result()

def main():
    args = sys.argv
    if len(args) < 3:
        print('> python segmentation.py {input directory} {output directory} {small window size = 3} {large window size = 17}')
        sys.exit(1)
    flist = np.sort(glob.glob(f'{args[1]}*.tif*'))
    print(f'Found {len(flist)} files')
    processFiles(flist,*args[2:])


if (__name__ == "__main__"):
    main()