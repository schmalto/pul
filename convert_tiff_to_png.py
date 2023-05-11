from osgeo import gdal
import glob
from tqdm import tqdm
import os
from PIL import Image 
    
options_list = [
    '-ot BYTE',
    '-of PNG',
    '-b 3',
    '-scale',
    "creationOptions=['PHOTOMETRIC=RGB', 'ALPHA=PREMULTIPLIED']"
]           

options_string = " ".join(options_list)
    
def gdal_convert():
    files = getAllTIFFFiles()
    filename = [("/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/satellitedata_converted/" + os.path.basename(item)) for item in files]
    for i in tqdm(range(len(files))):
        input = gdal.Open(files[i])
        gdal.Translate(
            filename[i].replace(".tif", ".jpeg"),
            input,
            format='JPEG',creationOptions=['PHOTOMETRIC=RGB', 'ALPHA=PREMULTIPLIED']
        )
    dirname = "/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/satellitedata_converted"
    items_in_dir = os.listdir(dirname)
    for item in tqdm(items_in_dir):
        if not item.endswith(".jpeg"):
            os.remove(os.path.join(dirname, item))



def getAllTIFFFiles():
    files = glob.glob("/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/satellitedata/**/*.tif", recursive=True)
    for item in tqdm(files):
        if not item.endswith(".tif"):
            files.remove(item)
    return files

def convert_pil():
    images = getAllTIFFFiles()
    filename = [("/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/satellitedata_converted/" + os.path.basename(item)) for item in images]
    
    for i in tqdm(range(len(images))):
        with Image.open(images[i]) as img:
            img.save(filename[i].replace(".tif", ".png"))


if __name__ == "__main__":
    gdal_convert()