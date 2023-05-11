import rasterio
from rasterio.plot import reshape_as_image
from rasterio.enums import ColorInterp
import numpy as np
import glob
import os
from tqdm import tqdm
import imageio
from rasterio.io import MemoryFile

class GeoTiffHelpers:
    def __init__(self):
    # Define the CMYK to RGB color conversion lookup table
        self.cmyk2rgb = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            self.cmyk2rgb[i, 0] = 255 - i
            self.cmyk2rgb[i, 1] = 255 - i
            self.cmyk2rgb[i, 2] = 255 - i

    def __getAllTIFFFiles(self, folder, recursiveSearch=True):
        files = glob.glob(folder + "**/*.tif", recursive=recursiveSearch)
        for item in files:
            if not item.endswith(".tif"):
                files.remove(item)
        return files

    def convert_cmyk_tiff_to_png(self, inputname, outputname):
    # Open the GeoTIFF file
        with rasterio.open(inputname) as src:

            # Read the bands from the GeoTIFF file
            bands = src.read()

            profile = src.profile

            # Convert the CMYK bands to RGB using the lookup table
            rgb_bands = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
            for i in range(bands.shape[0]):
                if src.colorinterp[i] == ColorInterp.alpha:
                    continue
                cmyk_vals = bands[i].reshape(-1, 1)
                rgb_vals = self.cmyk2rgb[cmyk_vals.flatten()]
                rgb_vals = rgb_vals.reshape(bands.shape[1], bands.shape[2], 3)
                rgb_bands[:, :, i % 3] += rgb_vals[:, :, i % 3]

        # Invert the RGB image
        rgb_bands = 255 - rgb_bands
        profile.update(dtype=rasterio.uint8, count=3)

        with rasterio.open(outputname, 'w', **profile) as dst:
            dst.write(rgb_bands.transpose((2, 0, 1)))

    def convert_rgb_tiff_to_png(self, inputname, outputname):
        with rasterio.open(inputname) as src:
            red = src.read(3)
            green = src.read(2)
            blue = src.read(1)
            merged = np.dstack((red, green, blue))
        merged = (merged / merged.max()) * 255
        merged = merged.astype(np.uint8)

        with MemoryFile() as memfile:
            with memfile.open(driver='PNG', width=merged.shape[2], height=merged.shape[1], count=3, dtype='uint8') as dst:
                dst.write(merged.transpose(2, 0, 1))
                imageio.imwrite(outputname, dst.read())

    def folder_convert_tiff_to_png(self, inputfolder, outputfolder, removeAllNonPNGFiles=False):
        files = self.__getAllTIFFFiles(inputfolder)
        filename = [outputfolder + os.path.basename(item) for item in files]
        for i in tqdm(range(len(files))):
            #self.convert_tiff_to_png(files[i], filename[i].replace(".tif", ".png"))
            self.convert_cmyk_tiff_to_png(files[i], filename[i].replace(".tif", ".png"))
        if removeAllNonPNGFiles:
            items_in_dir = os.listdir(outputfolder)
            for item in items_in_dir:
                if not item.endswith(".png"):
                    os.remove(os.path.join(outputfolder, item))

    def showTIFF(self, filename):
        # Open the GeoTIFF image
        dataset = rasterio.open(filename)
        rasterio.plot.show(dataset)
        dataset.close()

if __name__ == "__main__":
    geotiff = GeoTiffHelpers()
    #geotiff.convert_tiff_to_png("example.tif", "example2.png")
    geotiff.folder_convert_tiff_to_png("/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/Kraichgau_Nord_20220725_20210326_psscene_visual/", "/home/tobias/01_Uni/Stuttgart/SoSe23/PÜL/labelTest/", removeAllNonPNGFiles=True)


