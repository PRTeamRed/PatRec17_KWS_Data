"""
Using DIVAServices and the there provided Otsu binarization to binarize the images.
"""
import pickle
from pathlib import Path

from divaServices import ExecuteOnDivaServices
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy
from PIL import Image, ImageDraw
from svgpathtools import svg2paths, parser

imgPath = "images/"
binUrl = "http://divaservices.unifr.ch/api/v2/binarization/otsubinarization/1"
binPath = imgPath + "bin/"
cropPath = imgPath + "crop/"
svgPath = "ground-truth/locations/"
serialized_name = cropPath + "serializedWords"


def createDir(directory):
    if not exists(directory):
        makedirs(directory)


def main():
    createDir(binPath)
    createDir(cropPath)

    # binarize()

    # gets list of words for each image (each word is a numpy array)
    words = cropImages()


def binarize():
    print("Binarizing images...")
    divaService = ExecuteOnDivaServices()

    for img in getImageNames():
        divaService.main(binUrl, img, imgPath, binPath)


def getImageNames():
    return [f for f in listdir(imgPath) if isfile(join(imgPath, f))]


def cropImages():
    print("Cropping images...")
    if Path(serialized_name).is_file():
        return pickle.load(open(serialized_name, "rb"))

    words = []
    count = 0
    for img in getImageNames():
        if count < 2:
            createDir(cropPath + img.replace(".jpg", ""))
            words.append(cropImage(img.replace(".jpg", "")))
        count += 1
    pickle.dump(words, open(serialized_name, "wb"))
    return words


def cropImage(imgNumber):
    polygons = getPolygonsFromSVGPaths(imgNumber)
    words = []
    for i in range(len(polygons)):
        words.append(cropWord(polygons[i], imgNumber, i))
    return words


def getPolygonsFromSVGPaths(imgNumber):
    paths, attributes = svg2paths(svgPath + imgNumber + ".svg")
    polygons = []
    for k, v in enumerate(attributes):
        d = v['d']
        # remove letters
        d = d.replace("M", "")
        d = d.replace("L", "")
        d = d.replace("Z", "")
        it = iter([float(coord) for coord in d.split()])  # convert to floats
        polygon = list(zip(it, it))  # create polygon
        polygons.append(polygon)
    return polygons


def cropWord(polygon, imgNumber, croppedImgNumber):
    # inspired by https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil

    # read image as RGB and add alpha (transparency)
    im = Image.open("images/" + imgNumber + ".jpg").convert("RGBA")

    # convert to numpy (for convenience)
    imArray = numpy.asarray(im) / 255

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape, dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :, :3] = imArray[:, :, :3]

    # transparency (4th column)
    newImArray[:, :, 3] = mask * 255

    # TODO cropped image file same size as input image

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")

    newIm.save(cropPath + str(imgNumber) + "/crop_" + str(croppedImgNumber) + ".png")

    return newImArray


if __name__ == "__main__":
    main()
