from PIL import Image
from collections import defaultdict
import pprint
import glob
import os
filepath = 'test_data/test'

def preprocess():
    for filename in glob.glob('test_data/test/tomato/*.jpg'):
        img = Image.open(filename)
        width, height = img.size
        if(width != 300 or height != 300):
            print('width: ' + str(width) + 'height: ' + str(height))
            resizedImg = img.resize((300,300), Image.ANTIALIAS)
            newWidth, newHeight = resizedImg.size
            print('new size: ' + str(newWidth) + 'height: ' + str(newHeight))
            print(filename)
            resizedImg.save(filename)

    for filename in glob.glob('test_data/test/cherry/*.jpg'):
        img = Image.open(filename)
        width, height = img.size
        if(width != 300 or height != 300):
            print('width: ' + str(width) + 'height: ' + str(height))
            resizedImg = img.resize((300,300), Image.ANTIALIAS)
            newWidth, newHeight = resizedImg.size
            print('new size: ' + str(newWidth) + 'height: ' + str(newHeight))
            print(filename)
            resizedImg.save(filename)

    for filename in glob.glob('test_data/test/strawberry/*.jpg'):
        img = Image.open(filename)
        width, height = img.size
        if(width != 300 or height != 300):
            print('width: ' + str(width) + 'height: ' + str(height))
            resizedImg = img.resize((300,300), Image.ANTIALIAS)
            newWidth, newHeight = resizedImg.size
            print('new size: ' + str(newWidth) + 'height: ' + str(newHeight))
            print(filename)
            resizedImg.save(filename)


# def convertGrayScale():
#     for filename in glob.glob('data/train_data/strawberry/*.jpg'):
#         if(is_gray_scale(filename)):
#             img = Image.open(filename)
#             rgbimg = Image.new('RGB', img.size)
#             rgbimg.paste(img)
#             print(filename)
#             rgbimg.save(filename)
#
#     for filename in glob.glob('data/train_data/cherry/*.jpg'):
#         if(is_gray_scale(filename)):
#             img = Image.open(filename)
#             rgbimg = Image.new('RGB', img.size)
#             rgbimg.paste(img)
#             print(filename)
#             rgbimg.save(filename)
#
#     for filename in glob.glob('data/train_data/tomato/*.jpg'):
#         if(is_gray_scale(filename)):
#             img = Image.open(filename)
#             rgbimg = Image.new('RGB', img.size)
#             rgbimg.paste(img)
#             print(filename)
#             rgbimg.save(filename)
#
#
#
#
# def is_gray_scale(filename):
#     img = Image.open(filename).convert('RGB')
#     width, height = img.size
#     for i in range(width):
#         for j in range(height):
#             r, g, b = img.getpixel((i, j))
#             if r != g != b:
#                 return False
#     return True

if __name__ == '__main__':
    preprocess()
    print("Preprocess complete")
    # convertGrayScale()