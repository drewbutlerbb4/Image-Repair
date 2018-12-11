"""

In the following code Image refers to PIL.Image
"""
from PIL import Image, ImageDraw
import os
import random


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


# TODO: Implement Fade
def crease_image(image, crease_size, use_fade):
    """
    Adds a crease to the given image


    :param image:       The original image to be creased
    :param crease_size: The pixel width of the crease
    :param use_fade:    Whether or not to fade out the crease
    :return:            The corresponding image with a crease
    """

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
    return_image = Image.new(image.mode, image.size)
    return_image.putdata(image_to_list(image))
    draw = ImageDraw.Draw(return_image)
    draw.line((line_x,line_y, line_x2,line_y2), width=crease_size)

    return return_image


# TODO: Implement fade
def blotch_image(image, blotch_size, use_fade):

    return_image = Image.new(image.mode, image.size)
    return_image.putdata(image_to_list(image))
    draw = ImageDraw.Draw(return_image)
    print(((2 * blotch_size) + image.size[0]) - blotch_size)
    print(((2 * blotch_size) + image.size[1]) - blotch_size)
    x1 = random.randint(0, blotch_size + image.size[0]) - blotch_size
    y1 = random.randint(0, blotch_size + image.size[1]) - blotch_size
    x2 = x1 + blotch_size
    y2 = y1 + blotch_size
    print(x1,y1,x2,y2)
    draw.ellipse([x1,y1, x2,y2], fill='white')
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

"""
# Helpful Code
im = get_image("../dataset/toy-set/soldiers.jpg")
im2 = blotch_image(im,100,False)
im2.show()


im = get_image("../dataset/toy-set/soldiers.jpg")
print(im.size)
draw = ImageDraw.Draw(im)
draw.line((0,0, 1023,500),width=6)
list2 = image_to_list(im)
im2 = list_to_image(list2, im.mode, im.size)
im2.show()
"""