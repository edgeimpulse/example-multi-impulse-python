import numpy as np
import cv2

# Extracted from Edge Impulse Linux Python SDK
# https://github.com/edgeimpulse/linux-sdk-python/blob/2f69d12eb0641fed901ec59bc1af554db60c6ac3/edge_impulse_linux/image.py#L146

def resize_with_letterbox(image, target_width, target_height):
    """Resize an image while maintaining aspect ratio using letterboxing.

    Args:
        image: The input image as a NumPy array.
        target_size: A tuple (width, height) specifying the desired output size.

    Returns:
        The resized image as a NumPy array and the letterbox dimensions.
    """

    height, width = image.shape[:2]

    # Calculate scale factors to preserve aspect ratio
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y)

    # Calculate new dimensions and padding
    new_width = int(width * scale)
    new_height = int(height * scale)
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad

    # Resize image and add padding
    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def get_features_from_image_with_studio_mode(img, mode, output_width, output_height, is_grayscale):
    """
    Extract features from an image using different resizing modes suitable for Edge Impulse Studio.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        mode (str): The resizing mode to use. Options are 'fit-shortest', 'fit-longest', and 'squash'.
        output_width (int): The desired output width of the image.
        output_height (int): The desired output height of the image.
        is_grayscale (bool): Whether the output image should be converted to grayscale.

    Returns:
        tuple: A tuple containing:
            - features (list): A list of pixel values in the format (R << 16) + (G << 8) + B for color images,
              or (P << 16) + (P << 8) + P for grayscale images.
            - resized_img (numpy.ndarray): The resized image as a NumPy array.
    """
    features = []

    in_frame_cols = img.shape[1]
    in_frame_rows = img.shape[0]

    if mode == 'fit-shortest':
        aspect_ratio = output_width / output_height
        if in_frame_cols / in_frame_rows > aspect_ratio:
            # Image is wider than target aspect ratio
            new_width = int(in_frame_rows * aspect_ratio)
            offset = (in_frame_cols - new_width) // 2
            cropped_img = img[:, offset:offset + new_width]
        else:
            # Image is taller than target aspect ratio
            new_height = int(in_frame_cols / aspect_ratio)
            offset = (in_frame_rows - new_height) // 2
            cropped_img = img[offset:offset + new_height, :]

        resized_img = cv2.resize(cropped_img, (output_width, output_height), interpolation=cv2.INTER_AREA)
    elif mode == 'fit-longest':
        resized_img = resize_with_letterbox(img, output_width, output_height)
    elif mode == 'squash':
        resized_img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if is_grayscale:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        pixels = np.array(resized_img).flatten().tolist()

        for p in pixels:
            features.append((p << 16) + (p << 8) + p)
    else:
        pixels = np.array(resized_img).flatten().tolist()

        for ix in range(0, len(pixels), 3):
            r = pixels[ix + 0]
            g = pixels[ix + 1]
            b = pixels[ix + 2]
            features.append((r << 16) + (g << 8) + b)

    return features, resized_img