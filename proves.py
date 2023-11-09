import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, exposure, transform

# Load the image from your file system
image = io.imread("Practica1/dat/a2/data/train/bedroom/image_0001.jpg", as_gray=True)

# Resize the image to 200x200
image = transform.resize(image, (200, 200), anti_aliasing=True)

# Compute HOG features
fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(10, 10),
    cells_per_block=(1, 1),
    visualize=True,
    block_norm='L2-Hys'  # You can add this parameter for better normalization
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
