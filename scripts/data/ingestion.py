'''
TODO
'''

import argparse
import cv2
import json
from matplotlib import pyplot
import numpy
from pathlib import Path


SIZE = (2048, 2488)

CLASSES = {
    "vine": 1,
    "trunk": 2,
    "post": 3,
    "leaf": 4,
    "sign": 5,
}

def main(args):
    image = numpy.ones(SIZE) * -1

    labeldata = json.load(args.json_file.open("r"))
    shapes = labeldata["shapes"]
    for shape in shapes:
        assert shape["shape_type"] == "polygon"
        classid = CLASSES[shape["label"].lower()]
        altered = image.copy()
        # TODO: Treat the ints better
        cv2.fillPoly(
            img=altered,
            pts=[numpy.array(shape["points"], dtype=int)],
            color=classid,
        )
        # Check that we haven't altered an existing real label
        already_mask = image > 0
        assert numpy.all(image[already_mask] == altered[already_mask])
        # Then update our understanding
        image = altered

    # Make everything else background
    image[image < 0] = 0
    # And save
    image = image.astype(numpy.uint8)
    cv2.imwrite(str(args.output_path), image)

    # Then make a debug version
    pyplot.imsave(str(args.output_path).replace(".png", "_vis.png"), image)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # TODO: How many input types will there be? Re-organize
    parser.add_argument(
        "-j", "--json-file",
        help="Polygonally labelled JSON file.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o", "--output-path",
        help="Where the annotated image should be written.",
        required=True,
        type=Path,
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
