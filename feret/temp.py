import os, glob
import feret
import tifffile as tif

path = r"D:\Arbeit\nPSize\Auswertungen\Biypramiden_FEI_Nov_2020\Results"

for i, file in enumerate(glob.glob(os.path.join(path, '*_binary.tif'))):
    # if i != 22: continue
    img = tif.imread(file)
    # print(file)
    res = feret.calc(img)


