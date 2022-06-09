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

    if args.file_type == "diffgram-json":
        image = diffgram_label(image, args.json_file)
    elif args.file_type == "coco-json":
        image = coco_label(image, args.json_file)
    else:
        raise NotImplementedError()

    # Make everything else background
    image[image < 0] = 0
    # And save
    image = image.astype(numpy.uint8)
    cv2.imwrite(str(args.output_path), image)

    # Then make a debug version
    pyplot.imsave(str(args.output_path).replace(".png", "_vis.png"), image)


def diffgram_label(image, json_file):
    '''
    Process the json files as they come out of diffgram exports. (I think, it's
    the JSON type that Hussain shared with me first anyway.)
    '''
    labeldata = json.load(json_file.open("r"))
    shapes = labeldata["shapes"]
    for shape in shapes:
        assert shape["shape_type"] == "polygon"
        classid = CLASSES[shape["label"].lower()]
        image = draw_polygon(image, classid, shape["points"])
    return image


def coco_label(image, json_file):
    '''Process files as they come out of CVAT using the COCO format.'''
    labeldata = json.load(json_file.open("r"))
    annotations = labeldata["annotations"]
    for annotation in annotations:
        classid = annotation["category_id"]
        image = draw_polygon(
            image,
            classid,
            # (x, y) points all come out interleaved as [x1, y1, x2, y2, ...]
            numpy.array(annotation["segmentation"]).reshape((-1, 2)),
        )
    return image


def draw_polygon(image, classid, points):
    altered = image.copy()
    # TODO: Treat the ints better
    cv2.fillPoly(
        img=altered,
        pts=[numpy.array(points, dtype=int)],
        color=classid,
    )
    # Check that we haven't altered an existing real label
    already_mask = image > 0
    # assert numpy.all(image[already_mask] == altered[already_mask])
    # Then update our understanding
    return altered


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
    parser.add_argument(
        "-t", "--file-type",
        help="Choose from a limited set of options for ingestible filetypes.",
        required=True,
        choices=["diffgram-json", "coco-json"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
