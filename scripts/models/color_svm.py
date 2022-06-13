'''
Trains an SVM model based on pixel color. Because doing this on all pixels
would be crazy and prohibitively expensive (according to the SVM docs more than
10,000 data points is pushing things and each image has 5 million data points)
there is a mechanism to downsample images by class. The cost grows
quadratically with data size, apparently.

The final output mechanism is still TODO depending on how experiments are run.
'''

import argparse
from cProfile import Profile
import cv2
from joblib import dump, load
import logging
from matplotlib import pyplot
import numpy
from pathlib import Path
from sklearn import svm
import time


# Modes, we can either treat a pixel as a single 3-element vector (RGB) or
# treat it as a HxWx3 vector of RGBRGBRGB... in the area around it
SINGLE = "single"
AREA = "area"
# Square radius in pixels (center-radius:center+radius+1)
AREA_RADIUS = 3

CLASSES = [0, 1, 2, 3, 4, 5]


def main(datapath, mode, number, savedir):

    logging.info("Loading data...")
    data, labels = load_data(datapath, mode, number)

    logging.info(f"Training SVC with {data.shape} data, {labels.shape} labels...")
    classifier = svm.SVC()
    classifier.fit(data, labels)

    savepath = savedir.joinpath(f"svm_{int(time.time() * 1e6)}.pth")
    logging.info(f"Saving model to {savepath}")
    dump(classifier, savepath)


def load_data(datapath, mode, number):
    '''
    We want to load images from the cityscapes format because that's what we
    are already working with for mmsegmentation code. Randomly sample a certain
    number per class per image.

    Arguments:
        datapath: Base cityscapes format folder from which to draw images
        number: Number that we want to sample from each class, per picture

    Returns: two-element tuple of numpy arrays:
        [0]: size (N, X) array of data, where X is 3 or 3xHxW depending on mode
        [1]: size (N,) array of class labels corresponding to that data
    '''

    imgs = datapath.joinpath("img_dir", "train")
    anns = datapath.joinpath("ann_dir", "train")
    data = []
    labels = []
    for i, (imgpath, annpath) in enumerate(
                zip(sorted(imgs.glob("*png")),
                    sorted(anns.glob("*png")))
            ):

        if i % 20 == 0:
            logging.info(f"Loading {imgpath.name}, {annpath.name}")

        img = cv2.imread(str(imgpath), cv2.IMREAD_UNCHANGED)
        ann = cv2.imread(str(annpath), cv2.IMREAD_UNCHANGED)
        for classid in CLASSES:
            # argwhere is the time sink
            pixels = numpy.argwhere(ann == classid)
            # Skip the cases where there were none of that class
            if len(pixels) == 0:
                continue

            # randint could conceivably lead to some double-samples, but the
            # sample numbers are so low I think that's okay. It's much faster
            # than random.choice(range())
            if pixels.shape[0] <= number:
                indices = range(pixels.shape[0])
            else:
                indices = numpy.random.randint(0, pixels.shape[0], size=number)

            for index in indices:
                px = pixels[index]
                if mode == SINGLE:
                    data.append(img[px[0], px[1]])
                    labels.append(ann[px[0], px[1]])
                elif mode == AREA:
                    vector = img[
                        px[0]-AREA_RADIUS:px[0]+AREA_RADIUS+1,
                        px[1]-AREA_RADIUS:px[1]+AREA_RADIUS+1
                    ].flatten()
                    # If the vector isn't the right length it may be because
                    # the pixel was sampled along the image edge. Just skip it.
                    if len(vector) == 3 * (2 * AREA_RADIUS + 1)**2:
                        data.append(vector)
                        labels.append(ann[px[0], px[1]])
                else:
                    raise NotImplementedError()

    return numpy.array(data), numpy.array(labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--data-path",
        help="Base folder in the cityscapes format from which to draw images.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-m", "--mode",
        help="Choose between single-pixel and area-based classification.",
        default=SINGLE,
        choices=[SINGLE, AREA],
    )
    parser.add_argument(
        "-n", "--number-per-class",
        help="Number of pixels to randomly select per-class, per image."
             " Negative number means take all pixels.",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-s", "--save-dir",
        help="Directory in which to save the output model.",
        default=Path("/tmp/"),
        type=Path,
    )
    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()
    main(datapath=args.data_path,
         mode=args.mode,
         number=args.number_per_class,
         savedir=args.save_dir,
         )
