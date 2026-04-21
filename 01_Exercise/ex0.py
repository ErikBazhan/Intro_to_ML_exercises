import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # ToDo: Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        self._image: np.ndarray = np.zeros(0)
        
        if self._image is None:
            raise FileNotFoundError(f"Image could not be loaded: {image_path}")
        
        if self._colour_type == "RGB":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        elif self._colour_type == "Gray":
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Show the image depending on the colour type.
        if self._colour_type == "BGR":
            plt.imshow(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
        elif self._colour_type == "RGB":
            plt.imshow(self._image)
        elif self._colour_type == "Gray":
            plt.imshow(self._image, cmap="gray")

        plt.axis("off")
        plt.show()

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # ToDo: Save the image.
        if self._image is None or self._image.size == 0:
            raise ValueError("No image loaded to save.")

        if self._colour_type == "RGB":
            image_to_save = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = self._image

        image_saved = cv2.imwrite(total_image_path, image_to_save)

        if not image_saved:
            raise IOError(f"Image could not be saved: {total_image_path}")

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # ToDo: Perform the colour conversion.
        self._image = self._image[:, :, ::-1]

        # ToDo: Update the colour type.
        if self._colour_type == "BGR":
            self._colour_type = "RGB"
        else:
            self._colour_type = "BGR"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        # ToDo: Clip the image values to the given values.
        if clip_min < 0 or clip_max > 255 or clip_min > clip_max:
            raise ValueError("clip_min and clip_max must satisfy 0 <= min <= max <= 255.")

        self._image[self._image < clip_min] = clip_min
        self._image[self._image > clip_max] = clip_max

    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")
        
        image = self._image.astype(np.float32)

        if self._colour_type == "RGB":
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
        else:  # BGR
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]

        if method == "lightness":
            self._image = (
                (np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b))
                / 2
            ).astype(np.uint8)

        if method == "average":
            self._image = ((r + g + b) / 3).astype(np.uint8)

        if method == "luminosity":
            self._image = (0.21 * r + 0.72 * g + 0.07 * b).astype(np.uint8)

        self._colour_type = "Gray"

    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        # ToDo: Rotate the image depending on the given rotation value.
        rotation = degrees % 360

        if rotation == 0:
            return

        if self._image.ndim == 2:
            if rotation == 90:
                self._image = np.transpose(self._image, (1, 0))[:, ::-1]
            elif rotation == 180:
                self._image = self._image[::-1, ::-1]
            elif rotation == 270:
                self._image = np.transpose(self._image, (1, 0))[::-1, :]

        elif self._image.ndim == 3:
            if rotation == 90:
                self._image = np.transpose(self._image, (1, 0, 2))[:, ::-1, :]
            elif rotation == 180:
                self._image = self._image[::-1, ::-1, :]
            elif rotation == 270:
                self._image = np.transpose(self._image, (1, 0, 2))[::-1, :, :]

    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        if flip_value == 0:
            self._image = self._image[::-1, ...]
        elif flip_value == 1:
            self._image = self._image[:, ::-1, ...]
        else:  # flip_value == 2
            self._image = self._image[::-1, ::-1, ...]

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # ToDo: Check that the given parameters are valid!
        current_height, current_width = self._image.shape[:2]

        if new_height <= 0 or new_width <= 0:
            raise ValueError("new_height and new_width must be positive!")

        if new_height > current_height or new_width > current_width:
            raise ValueError("Crop size must not be larger than the image size!")

        # ToDo: Crop the image around the center.
        start_row = (current_height - new_height) // 2
        start_col = (current_width - new_width) // 2

        end_row = start_row + new_height
        end_col = start_col + new_width

        self._image = self._image[start_row:end_row, start_col:end_col]

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # ToDo: Resize the image. Research the available options in CV2.
        if new_height <= 0 or new_width <= 0:
            raise ValueError("new_height and new_width must be positive!")

        self._image = cv2.resize(
            self._image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
