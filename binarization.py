"""
Using DIVAServices and the there provided Otsu binarization to binarize the images.
"""

from divaServices import ExecuteOnDivaServices
from os import listdir
from os.path import isfile, join

imgPath = "images"
binUrl = "http://divaservices.unifr.ch/api/v2/binarization/otsubinarization/1"
binPath = "images/bin/"

imageNames = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]


divaService = ExecuteOnDivaServices()

for img in imageNames:
    divaService.main(binUrl, img, imgPath,  binPath)

