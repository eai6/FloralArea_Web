# #from image_slicer import Tile, slice as slice_image
# #from app.scripts import deeplab_model 
# from PIL import Image
# # import numpy as np
# # from numpy import asarray
# import os

# from cv import img_processing as ip


# def slice_image(input_image_path, output_directory, num_tiles):
#     """
#     Slice the input image into the specified number of tiles.

#     Parameters:
#     - input_image_path: Path to the input image file.
#     - output_directory: Directory to save the sliced tiles.
#     - num_tiles: Number of tiles to create.

#     Returns:
#     - List of paths to the sliced tiles.
#     """
#     # get extension of input image
#     extension = os.path.splitext(input_image_path)[1]
    
#     # Open the input image
#     img = Image.open(input_image_path)

#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     # Get image dimensions
#     width, height = img.size

#     # Calculate tile size
#     tile_width = width // num_tiles
#     tile_height = height // num_tiles

#     # List to store paths of sliced tiles
#     sliced_tiles = []

#     # Iterate over rows and columns to slice the image
#     for i in range(num_tiles):
#         for j in range(num_tiles):
#             # Calculate coordinates for cropping each tile
#             left = j * tile_width
#             upper = i * tile_height
#             right = left + tile_width
#             lower = upper + tile_height

#             # Crop the tile
#             tile = img.crop((left, upper, right, lower))

#             # Save the tile to the output directory
#             tile_path = os.path.join(output_directory, f"tile_{i}_{j}{extension}")
#             tile.save(tile_path)

#             # Append the path to the list
#             sliced_tiles.append(tile_path)

#     return sliced_tiles



# def combine_tiles(tiles_directory, output_image_path, num_tiles, component = 'reference_object'):
#     """
#     Combine sliced tiles into a single image.

#     Parameters:
#     - tiles_directory: Directory containing the sliced tiles.
#     - output_image_path: Path to save the combined image.
#     - num_tiles: Number of tiles per row and column.

#     Returns:
#     - Path to the combined image.
#     """
#     # Create a list to store the tiles
#     tiles = []

#     # get extension of input image
#     extension = os.listdir(tiles_directory)[0].split('.')[-1]
#     #print(extension)

#     # Load each tile and append to the list
#     for i in range(num_tiles):
#         for j in range(num_tiles):
#             #tile_path = os.path.join(tiles_directory, f"tile_{i}_{j}.{extension}")
#             tile_path = os.path.join(tiles_directory, f"tile_{i}_{j}.{extension}_{component}_mask.png")
#             tile = Image.open(tile_path)
#             tiles.append(tile)

#     # Calculate the dimensions of the combined image
#     tile_width, tile_height = tiles[0].size
#     total_width = tile_width * num_tiles
#     total_height = tile_height * num_tiles

#     # Create a new image with the calculated dimensions
#     combined_image = Image.new("RGB", (total_width, total_height))
#     # combine image is black and white
#     #combined_image = Image.new("L", (total_width, total_height))

#     # Paste each tile onto the combined image
#     for i in range(num_tiles):
#         for j in range(num_tiles):
#             left = j * tile_width
#             upper = i * tile_height
#             combined_image.paste(tiles[i * num_tiles + j], (left, upper))

#     # Save the combined image
#     combined_image.save(output_image_path)

#     return output_image_path


# # load image
# def runTile(yolo, threshold, component,  filename):
#     #im = Image.open(filename)
#     '''
#     return the are of the tile image
#     '''
#     result = yolo.runInference(filename, component, threshold)
#     mask = yolo.getMask(result)
#     reference_pixel_area = ip.getPixelArea(mask)

#     # save mask
#     #mask.save(f"api/data/tiles/{filename.split('/')[-1]}_{component}_mask.png")

#     # save plot
#     img = result[0].plot(conf=True, labels=False, boxes=True, masks=True)
#     #img = results[0].plot(conf=conf, labels=labels, boxes=boxes, masks=masks)
#     img = Image.fromarray(img[..., ::-1])
#     #filename = path.split("/")[-1]
#     filename1 = f"data/tiles/{filename.split('/')[-1]}_{component}_mask.png"
#     img.save(filename1)
    

#     return reference_pixel_area


# def runTilles(yolo, file_path, threshold, num_tiles = 2, component = 'reference_object'):
    
#     # slice the image
#     #num_tiles = 2
#     tiles = slice_image(file_path, 'data/tiles', num_tiles)

#     total_area = 0
#     # run model on each tile
#     for i in range(num_tiles):
#         for j in range(num_tiles):
#             tile_area  = runTile(yolo, threshold, component, tiles[i*num_tiles + j])
#             total_area += tile_area

#     # combine the tiles
#     filename = file_path.split('/')[-1]
#     # remove extension
#     filename = filename.split('.')[0]
#     filename = filename.split('@')[-1]
#     combined_image = combine_tiles('data/tiles', f'data/output/combined_mask_image.png', num_tiles, component)

#     #remove tiles
#     for tile in tiles:
#        os.remove(tile)

#     # return seg_image
#     return total_area, combined_image



from PIL import Image
import os
from cv import img_processing as ip


def slice_image(input_image_path, output_directory, num_tiles):
    """
    Slice the input image into the specified number of tiles.

    Parameters:
    - input_image_path: Path to the input image file.
    - output_directory: Directory to save the sliced tiles.
    - num_tiles: Number of tiles to create.

    Returns:
    - List of paths to the sliced tiles.
    """
    # Ensure absolute paths
    input_image_path = os.path.abspath(input_image_path)
    output_directory = os.path.abspath(output_directory)

    # Get extension of input image
    extension = os.path.splitext(input_image_path)[1]

    # Open the input image
    img = Image.open(input_image_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get image dimensions
    width, height = img.size

    # Calculate tile size
    tile_width = width // num_tiles
    tile_height = height // num_tiles

    # List to store paths of sliced tiles
    sliced_tiles = []

    # Iterate over rows and columns to slice the image
    for i in range(num_tiles):
        for j in range(num_tiles):
            # Calculate coordinates for cropping each tile
            left = j * tile_width
            upper = i * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # Crop the tile
            tile = img.crop((left, upper, right, lower))

            # Save the tile to the output directory
            tile_filename = f"tile_{i}_{j}{extension}"
            tile_path = os.path.join(output_directory, tile_filename)
            tile.save(tile_path)

            # Append the path to the list
            sliced_tiles.append(tile_path)

    return sliced_tiles


def combine_tiles(tiles_directory, output_image_path, num_tiles, component='reference_object'):
    """
    Combine sliced tiles into a single image.

    Parameters:
    - tiles_directory: Directory containing the sliced tiles.
    - output_image_path: Path to save the combined image.
    - num_tiles: Number of tiles per row and column.

    Returns:
    - Path to the combined image.
    """
    tiles_directory = os.path.abspath(tiles_directory)
    output_image_path = os.path.abspath(output_image_path)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # List to store tile images
    tiles = []

    # Find the correct file extension
    tile_files = [f for f in os.listdir(tiles_directory) if f.endswith("mask.png")]
    if not tile_files:
        raise FileNotFoundError("No mask images found in the tiles directory.")

    extension = tile_files[0].split('.')[-2]  # Extract extension before "_mask"

    # Load each tile and append to the list
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile_filename = f"tile_{i}_{j}.{extension}.png"
            tile_path = os.path.join(tiles_directory, tile_filename)
            if os.path.exists(tile_path):
                tile = Image.open(tile_path)
                tiles.append(tile)
            else:
                raise FileNotFoundError(f"Expected tile {tile_filename} not found.")

    # Calculate dimensions of the combined image
    tile_width, tile_height = tiles[0].size
    total_width = tile_width * num_tiles
    total_height = tile_height * num_tiles

    # Create a new image with calculated dimensions
    combined_image = Image.new("RGB", (total_width, total_height))

    # Paste each tile onto the combined image
    for i in range(num_tiles):
        for j in range(num_tiles):
            left = j * tile_width
            upper = i * tile_height
            combined_image.paste(tiles[i * num_tiles + j], (left, upper))

    # Save the combined image
    combined_image.save(output_image_path)

    return output_image_path


def runTile(yolo, threshold, component, filename):
    """
    Run the YOLO model on a single tile and return the area.
    """
    filename = os.path.abspath(filename)

    result = yolo.runInference(filename, component, threshold)
    mask = yolo.getMask(result)
    reference_pixel_area = ip.getPixelArea(mask)

    # Save plot
    img = result[0].plot(conf=True, labels=False, boxes=True, masks=True)
    img = Image.fromarray(img[..., ::-1])

    # Ensure output directory exists
    output_dir = os.path.abspath("data/tiles")
    os.makedirs(output_dir, exist_ok=True)

    # Save the mask image
    mask_filename = f"{os.path.basename(filename)}_{component}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    img.save(mask_path)

    return reference_pixel_area


def runTilles(yolo, file_path, threshold, num_tiles=2, component='reference_object'):
    """
    Slice an image, run the model on each tile, and recombine the segmented results.
    """
    file_path = os.path.abspath(file_path)

    # Ensure output directories exist
    tiles_dir = os.path.abspath("data/tiles")
    output_dir = os.path.abspath("data/output")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Slice the image
    tiles = slice_image(file_path, tiles_dir, num_tiles)

    total_area = 0
    for tile in tiles:
        total_area += runTile(yolo, threshold, component, tile)

    # Combine the tiles
    combined_image_path = os.path.join(output_dir, "combined_mask_image.png")
    try:
        combined_image = combine_tiles(tiles_dir, combined_image_path, num_tiles, component)
    except FileNotFoundError as e:
        print(f"Error combining tiles: {e}")
        # set combined_image to original input image
        combined_image = file_path

    # Remove tiles after processing
    for tile in tiles:
        os.remove(tile)

    return total_area, combined_image
