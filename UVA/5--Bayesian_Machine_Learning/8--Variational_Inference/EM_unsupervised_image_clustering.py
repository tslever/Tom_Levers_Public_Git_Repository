# Unsupervised Image Classification
# using Gaussian Mixture Model Expectation Maximization
# by William F. Basener
# University of Virginia, School of Data Science

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# read the image and convert to numpy array
im = Image.open('Haiti_Image.tif')
imArray = np.array(im)
nrows, ncols, nbands = np.shape(imArray)

# reshape the image array to num_observations by num_features (1,000,000 by 3)
X = np.reshape(imArray,[nrows*ncols,nbands])

# Create and train our Gaussian Mixture Expectation Maximization Model
model = GaussianMixture(n_components=4, tol=0.1)
model.fit(X)
# predict latent values
yhat = model.predict(X)

# reshape the result into an image
imSubset_hat = np.reshape(yhat,[nrows,ncols])

# plot the output, including the BIC in the title
plt.figure()
plt.suptitle('Unsupervised EM Classification, BIC = '+"{:e}".format(model.bic(X)))
plt.subplot(121)
plt.imshow(imArray)
plt.title('Color Image')
plt.subplot(122)
plt.imshow(imSubset_hat)
plt.title('GMM EM Classification Result')
plt.show()
