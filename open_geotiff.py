import rasterio
from rasterio.plot import show
import matplotlib

def showTIFF(filename):
    matplotlib.use('Qt5Agg')

    # Open the GeoTIFF image
    dataset = rasterio.open(filename)

    ax = show(dataset)
    print(ax)
    #fig.savefig('preview.png')

    # Close the dataset
    dataset.close()



with rasterio.open('20220726_160707_ssc15_u0001_visual_clip.tif') as src:

    # Print the number of bands
    print(f"Number of bands: {src.count}")