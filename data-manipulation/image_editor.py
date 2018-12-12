"""

In the following code Image refers to PIL.Image
"""
from PIL import Image, ImageDraw
import os
import random
import math


def get_image(image_path):
    """
    Returns the image at the given image_paht

    :param image_path:  A path to an image
    :return:            An image
    """

    return Image.open(image_path)


def image_to_list(image):
    """
    Returns a flattened list of the pixel values in the given image

    :param image:       An image
    :return:            Flattened list of pixels
    """

    return list(image.getdata())


def list_to_image(pixel_list, mode, size):
    """
    Returns an Image correspending to the image specifications given

    :param pixel_list:  A list of pixel values
    :param mode:        The image mode of the picture
    :param size:        The size of the image (tuple)
    :return:            An image
    """
    im = Image.new(mode, size)
    im.putdata(pixel_list)
    return im


def crease_image(image, crease_size, use_fade):
    """
    Adds a crease to the given image


    :param image:       The original image to be creased
    :param crease_size: The pixel width of the crease
    :param use_fade:    Whether or not to fade out the crease
    :return:            The corresponding image with a crease
    """

    if use_fade:
        orig_image = image
        image = Image.new(image.mode, image.size, '#000000')

    # Decides the two points on the edge of the image that will be the endpoints to the crease

    # The length of all four sides minus the four corners and one for array indexing
    max_edge = (2 * image.size[0]) + (2 * image.size[1]) - 5

    if random.randint(0, max_edge) < 2 * image.size[0] - 1:
        if random.randint(0,1) == 0:
            line_y = 0
        else:
            line_y = image.size[1] - 1
        line_x = random.randint(0, image.size[0] - 1)

        if random.randint(0, max_edge - image.size[0]) < image.size[0] - 1:
            line_y2 = abs(line_y - image.size[1] + 1)
            line_x2 = random.randint(0, image.size[0] - 1)
        else:
            if random.randint(0, 1) == 0:
                line_x2 = 0
            else:
                line_x2 = image.size[0] - 1
            line_y2 = random.randint(1, image.size[1] - 2)
    else:
        if random.randint(0,1) == 0:
            line_x = 0
        else:
            line_x = image.size[0] - 1
        line_y = random.randint(1, image.size[1] - 2)

        if random.randint(0, max_edge - image.size[1]) < image.size[1] - 1:
            line_x2 = abs(line_x - image.size[1] + 1)
            line_y2 = random.randint(1, image.size[1] - 2)
        else:
            if random.randint(0, 1) == 0:
                line_y2 = 0
            else:
                line_y2 = image.size[1] - 1
            line_x2 = random.randint(0, image.size[0] - 1)

    dx = line_x2 - line_x
    dy = line_y2 - line_y

    line_x -= dx
    line_y -= dy
    line_x2 += dx
    line_y2 += dy

    # Creates a copy image of the original and then overlays a line on it
    fade_image = Image.new(image.mode, image.size)
    fade_image.putdata(image_to_list(image))
    draw = ImageDraw.Draw(fade_image)

    return_image = Image.new(image.mode, image.size)

    if not use_fade:
        return_image.putdata(image_to_list(image))
        draw = ImageDraw.Draw(return_image)
        draw.line((line_x, line_y, line_x2, line_y2), fill='#ff0000', width=crease_size)
        return return_image

    if crease_size <= 2:
        draw.line((line_x, line_y, line_x2, line_y2), fill='#ff0000', width=crease_size)
    elif crease_size <= 5:
        draw.line((line_x, line_y, line_x2, line_y2), fill='#ff0001', width=crease_size)
        draw.line((line_x, line_y, line_x2, line_y2), fill='#ff0000', width=2)
    else:
        for x in range(0, crease_size - 2):
            fill_value = str((crease_size - 3) - x)
            if len(fill_value) > 7:
                raise IndexError("Too large of a crease size")
            while len(fill_value) < 4:
                fill_value = "0" + fill_value
            fill_color = "#ff" + fill_value
            draw.line((line_x, line_y, line_x2, line_y2), fill=fill_color, width=((crease_size - 3) - x))

    fade_data = image_to_list(fade_image)
    orig_data = image_to_list(orig_image)

    num_fades = crease_size - 2
    for pixel_iter in range(0, len(fade_data)):
        pixel_color = fade_data[pixel_iter]
        orig_pixel = orig_data[pixel_iter]
        if not pixel_color == (0,0,0):
            hex11 = math.floor((pixel_color[1] + 1) / 16)
            hex12 = (pixel_color[1] - (hex11 * 16))
            hex21 = math.floor((pixel_color[2] + 1) / 16)
            hex22 = (pixel_color[2] - (hex21 * 16))
            color_value = int(str(hex11) + str(hex12) + str(hex21) + str(hex22))
            frac = color_value / num_fades
            r_val = int(255 - ((255 - orig_pixel[0]) * frac))
            g_val = int(255 - ((255 - orig_pixel[1]) * frac))
            b_val = int(255 - ((255 - orig_pixel[2]) * frac))
            orig_data[pixel_iter] = (r_val, g_val, b_val)

    return_image.putdata(orig_data)
    return return_image


def blotch_image(image, blotch_size, use_fade):
    """
    Creates a blotch on the given image

    :param image:       The original image to be damaged
    :param blotch_size: The size of the blotch (in rectangular bounds)
    :param use_fade:    Whether or not the blotch should fade
    :return:            The image with an added blotch
    """

    if use_fade:
        orig_image = image
        image = Image.new(image.mode, image.size, '#000000')

    x1 = random.randint(0, blotch_size + image.size[0]) - blotch_size
    y1 = random.randint(0, blotch_size + image.size[1]) - blotch_size
    x2 = x1 + blotch_size
    y2 = y1 + blotch_size
    print(x1, y1, x2, y2)

    if (not use_fade) | (blotch_size <= 5):
        return_image = Image.new(image.mode, image.size)
        return_image.putdata(image_to_list(image))
        draw = ImageDraw.Draw(return_image)
        draw.ellipse([x1, y1, x2, y2], fill='white')
        return return_image

    draw = ImageDraw.Draw(image)
    if blotch_size % 2 == 0:
        jump = int((x2 - x1)/2)
        cb = ((jump + x1, jump + y1), (jump + x1 + 1, jump + y1 + 1))

        for x in range(0, int(blotch_size/2) - 1):
            fill_value = str(int((blotch_size/2)) - 2 - x)
            if len(fill_value) > 7:
                raise IndexError("Too large of a crease size")
            while len(fill_value) < 4:
                fill_value = "0" + fill_value
            fill_color = "#ff" + fill_value

            b = int(blotch_size/2) - x
            draw.ellipse([cb[0][0]-b, cb[0][1]-b, cb[1][0]+b, cb[1][1]+b], fill=fill_color)
    else:
        cb = (math.floor((x2 - x1)/2) + x1, math.floor((y2 - y1)/2) + y1)

        for x in range(0, int((blotch_size - 1)/2) - 1):
            fill_value = str(int((blotch_size - 1)/2 - 2 - x))
            if len(fill_value) > 7:
                raise IndexError("Too large of a crease size")
            while len(fill_value) < 4:
                fill_value = "0" + fill_value
            fill_color = "#ff" + fill_value

            b = int((blotch_size-1)/2) - x
            draw.ellipse([cb[0]-b,cb[1]-b, cb[0]+b,cb[1]+b], fill=fill_color)

    fade_data = image_to_list(image)
    orig_data = image_to_list(orig_image)

    num_fades = (blotch_size / 2) - 1 if blotch_size % 2 == 0 else ((blotch_size - 1) / 2) - 1
    for pixel_iter in range(0, len(fade_data)):
        pixel_color = fade_data[pixel_iter]
        orig_pixel = orig_data[pixel_iter]
        if not pixel_color == (0, 0, 0):
            hex11 = math.floor((pixel_color[1] + 1) / 16)
            hex12 = (pixel_color[1] - (hex11 * 16))
            hex21 = math.floor((pixel_color[2] + 1) / 16)
            hex22 = (pixel_color[2] - (hex21 * 16))
            color_value = int(str(hex11) + str(hex12) + str(hex21) + str(hex22))
            frac = color_value / num_fades
            r_val = int(255 - ((255 - orig_pixel[0]) * frac))
            g_val = int(255 - ((255 - orig_pixel[1]) * frac))
            b_val = int(255 - ((255 - orig_pixel[2]) * frac))
            orig_data[pixel_iter] = (r_val, g_val, b_val)

    return_image = Image.new(image.mode, image.size)
    return_image.putdata(orig_data)
    return return_image


# Untested
def preprocess_directory(data_path, label_path, damage_fn):
    """
    Preprocesses the data in data_path using the method damage_fn
    and stores the results in a new folder at label_path

    :param data_path:   Path to folder of image
    :param label_path:  Path to folder of our labels
    :param damage_fn:   A function that changes the given photo
    """

    file_names = os.listdir(data_path)
    os.mkdir(label_path)

    for file_name in file_names:
        file_path = data_path + "/" + file_name
        cur_label_path = label_path + "/" + file_name
        current_image = Image.open(file_path)
        label = damage_fn(current_image)
        label.save(cur_label_path, "JPEG")


def sample_damaging(image):
    """
    Sample damaging that does one blotch and one crease on every Image

    :param image:   The image to be damaged
    :return:        The damaged image
    """
    return crease_image(blotch_image(image, 100, False), 10, False)


# Test Code for damaging an image
im = get_image("../dataset/toy-set/soldiers.jpg")
im.show()
im = blotch_image(im, 100, True)
im = blotch_image(im, 100, True)
im = blotch_image(im, 100, True)
im = crease_image(im, 30, True)
im.show()
