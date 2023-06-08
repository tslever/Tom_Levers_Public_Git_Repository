import numpy as np
import pandas as pd
from PIL import Image

data_frame_of_classes_and_pixels = pd.read_csv('HaitiPixels.csv')
data_frame_of_classes_and_pixels = data_frame_of_classes_and_pixels.head(n = 63001)
data_frame_of_pixels = data_frame_of_classes_and_pixels[['Red', 'Green', 'Blue']]
data_frame_of_pixels = data_frame_of_pixels.astype(np.uint8)
array_of_pixels = data_frame_of_pixels.to_numpy()
shape_of_array_of_pixels = (251, 251, 3)
array_of_pixels = array_of_pixels.reshape(shape_of_array_of_pixels)
print(array_of_pixels)
image = Image.fromarray(array_of_pixels)
image.save('Image.png')