import numpy as np
from PIL import Image
from PIL import ImageOps


# def getPixelArea(img:Image) -> int:
#     '''
#     Get the number of white pixels in an image
#     '''
#     x,y,d = np.array(img).shape

#     count = 0
#     img_array = np.array(img)
#     for i in range(x):
#         for j in range(y):
#             #if img_array[i][j][0] == 255:
#             if img_array[i][j][0] > 0:
#                 count += 1
#             else:
#                 continue

#     return count

def crop_image(image, detections):
    """
    Crop a subset of the image using the given detections.

    Parameters:
        image (PIL.Image.Image): The input image.
        detections (list): List of detection coordinates in the format [x, y, w, h].

    Returns:
        PIL.Image.Image: The cropped image.
    """
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Extract the detection coordinates
    x, y, w, h = detections

    # Crop the image using the detection coordinates
    cropped_image = image_array[int(y-(h/2)): int(y+(h/2)), int(x-(w/2)): int(x+ (w/2))]

    # Convert the cropped image back to PIL Image
    cropped_image = Image.fromarray(cropped_image)

    return cropped_image



def getPixelArea(image):
    """
    Counts all the non-black pixels in a PIL Image.

    Parameters:
        image (PIL.Image.Image): The input image.
    
    Returns:
        int: The count of non-black pixels.
    """
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    
    # Check if the image is grayscale or RGB
    if len(image_array.shape) == 2:  # Grayscale image
        non_black_pixels = np.sum(image_array > 0)
    elif len(image_array.shape) == 3:  # RGB image
        # Sum across color channels to detect non-black pixels
        non_black_pixels = np.sum(np.any(image_array > 0, axis=-1))
    else:
        raise ValueError("Invalid image format. Expected grayscale or RGB image.")
    
    return non_black_pixels


def apply_preprocessing(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image to 640x640
    resized_image = image.resize((640, 640))
    
    # Apply contrast stretching
    contrast_stretched_image = ImageOps.autocontrast(resized_image)
    #contrast_stretched_image = resized_image
    
    # Save the image
    contrast_stretched_image.save(image_path)
    
    return contrast_stretched_image