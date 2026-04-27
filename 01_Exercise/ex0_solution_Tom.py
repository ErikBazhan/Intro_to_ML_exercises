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

        if colour_type == "Gray":
            self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self._image is None:
                raise FileNotFoundError(f"Could not load image from path: {image_path}")

        else:
            self._image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self._image is None:
                raise FileNotFoundError(f"Could not load image from path: {image_path}")
            if colour_type == "RGB":
                self._image = self._image[:, :, ::-1]

    def get_image_data(self):
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Show the image depending on the colour type.
        image_to_show = self._image

        if self._colour_type == "RGB":
            image_to_show = self._image[:, :, ::-1]

        cv2.imshow("Image", image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        image_to_save = self._image

        if self._colour_type == "RGB":
            image_to_save = self._image[:, :, ::-1]

        success = cv2.imwrite(total_image_path, image_to_save)
        if not success:
            raise IOError(f"Could not save image to: {total_image_path}")
        pass


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
        pass

        # ToDo: Update the colour type.
        if self._colour_type == "RGB":
            self._colour_type = "BGR"
        else:
            self._colour_type = "RGB"
        pass

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
        pass

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
        if self._colour_type == "BGR":
            r = self._image[:, :, 2]
            g = self._image[:, :, 1]
            b = self._image[:, :, 0]
        else:
            r = self._image[:, :, 0]
            g = self._image[:, :, 1]
            b = self._image[:, :, 2]

        if method == "lightness":
            self._image = (np.maximum(r,g,b) + np.minimum(r,g,b))/2

        if method == "average":
            self._image = (r+g+b)/3

        if method == "luminosity":
            self._image = (0.21 * r + 0.72 * g + 0.07 * b)

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
            axes = (1, 0) # Change HxW = WxH
        elif self._image.ndim == 3:
            axes = (1, 0, 2) # HxWxC = WxHxC
        else:
            raise ValueError("Unsupported image dimensions!")

        if rotation == 90:
            self._image = np.transpose(self._image, axes)[:, ::-1, ...] # reverse rows (:: -1)
        elif rotation == 180:
            self._image = self._image[::-1, ::-1, ...] # flip
        elif rotation == 270:
            self._image = np.transpose(self._image, axes)[::-1, :, ...] # reverse columns (::-1,)


    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        match flip_value:
            case 0:
                self._image = self._image[::-1, :, ...]
            case 1:
                self._image = self._image[:, ::-1, ...]
            case 2:
                self._image = self._image[::-1, ::-1, ...]
        pass

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
            raise ValueError(f"Crop size must not be larger than the image size! (Original size: Height = {current_height} ----- Width = {current_width})")


        # ToDo: Crop the image around the center.

        start_y = (current_height - new_height) // 2
        start_x = (current_width - new_width) // 2

        end_y = start_y + new_height
        end_x = start_x + new_width

        self._image = self._image[start_y:end_y, start_x:end_x, ...]
        pass

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
