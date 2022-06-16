'''
TODO
'''

import argparse
import cv2
import json
from matplotlib import pyplot
import numpy
from pathlib import Path


SIZE = (2048, 2448)

CLASSES = {
    "vine": 1,
    "trunk": 2,
    "post": 3,
    "leaf": 4,
    "sign": 5,
}

def main(args):

    image = numpy.ones(SIZE) * -1

    vis_image = None
    if args.file_type == "coco-json":
        image = coco_label(image, args.json_file)
    elif args.file_type == "diffgram-json":
        image, vis_image = diffgram_label(image, args.json_file)
    elif args.file_type == "hussain-json":
        image = hussain_json_label(image, args.json_file)
    else:
        raise NotImplementedError()

    # Make everything else background
    image[image < 0] = 0
    # And save
    image = image.astype(numpy.uint8)
    cv2.imwrite(str(args.output_path), image)

    # Then make a debug version
    if vis_image is None:
        vis_image = image
    pyplot.imsave(str(args.output_path).replace(".png", "_vis.png"), vis_image)


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


# This is horrible, but for the first diffgram export I had to manually go
# through and separate merged instances :'(
UNMERGED = {
    "3400_diffgram_annotations__source_task_98991_datetime_2022-06-02T23-53-19.191400.json": [0, 92, 275, 341, 520, 559, 746, 975, 1114, 1228, 1289, 1335, 1467, 1507, 1547, 1572, 1594, 1619, 1631, 1683, 1758, 1799, 1879, 2009, 2042, 2140, 2249, 2335, 2354, 2370, 2411, 2439, 2460, 2534, 2594, 2665, 2693, 2727, 2759, 2885, 2930, 2963, 2981, 3019, 3091, 3105, 3113, 3159, 3195, 3276, 3328, 3407, 3439, 3446, 3513, 3579, 3605, 3635, 3760, 3820, 3825, 3866, 3948, 3989, 4123, 4135, 4202, 4239, 4263, 4323, 4390, 4439, 4477, 4491, 4499, 4650, 4720, 4813, 4843, 4854, 4901, 4951, 5141, 5149, 5193, 5243, 5257, 5264, 5288, 5303, 5345, 5391, 5533, 5561, 5699, 5749, 5815, 5831, 5847, 5867, 5906, 5971, 5997, 6029, 6046, 6093, 6169, 6183, 6228, 6460, 6508, 6545, 6630, 6646, 6669, 6737, 6786, 6803, 6853, 6866, 6912, 6936, 6997, 7050, 7167, 7265, 7319, 7389, 7513, 7533, 7543, 7583, 7615, 7718, 7781]
}


def diffgram_label(image, json_file):
    '''Process the json files as they come out of Diffgram.'''
    # Extra debug image
    vis_image = image.copy()
    vis_value = max(CLASSES.values()) + 1

    labeldata = json.load(json_file.open("r"))
    label_map = labeldata["label_map"]
    # Identify which variable has the "instance_list" attribute. It appears to
    # be a random number in my first example so I'm not sure how stable that is
    instances = None
    for dictionary in labeldata.values():
        if "instance_list" in dictionary:
            # Make sure this only happens once
            assert instances is None
            instances = dictionary["instance_list"]
    # Then loop through each polygon
    for i, instance in enumerate(instances):
        assert instance["type"] == "polygon"
        classid = CLASSES[label_map[str(instance["label_file_id"])].lower()]
        points = numpy.array([(pt["x"], pt["y"]) for pt in instance["points"]])
        # Oh god so hacky. In my first diffgram image there appears to be a
        # big problem where I merged instances to make it exportable, and that
        # really screws up the rendering. I can't see how to detect that case,
        # so for now I'll detect it with number of points. Then I guess I'll
        # split up instances based on nearness of points? Ugh.
        if len(points) < 7500:
            image = draw_polygon(image, classid, points)
            vis_image = draw_polygon(vis_image, classid, points, add_points=vis_value)
        else:
            if json_file.name in UNMERGED:
                for subset in get_point_subsets(points, UNMERGED[json_file.name]):
                    image = draw_polygon(image, classid, subset)
                    vis_image = draw_polygon(vis_image, classid, subset, add_points=vis_value)
            else:
                discover_subsets(points)
    return image, vis_image


def hussain_json_label(image, json_file):
    '''
    Process the json files as they come out of whatever Hussain shared with me
    first.
    '''
    labeldata = json.load(json_file.open("r"))
    shapes = labeldata["shapes"]
    for shape in shapes:
        assert shape["shape_type"] == "polygon"
        classid = CLASSES[shape["label"].lower()]
        image = draw_polygon(image, classid, shape["points"])
    return image


def draw_polygon(image, classid, points, add_points=None):
    altered = image.copy()
    # TODO: Treat the ints better
    points = numpy.array(points, dtype=int)
    cv2.fillPoly(img=altered, pts=[points], color=classid)
    if add_points is not None:
        altered[points.T[1], points.T[0]] = add_points
        last = points[-1]
        altered[last[1]-2:last[1]+2, last[0]-2:last[0]+2] = add_points
    # Check that we haven't altered an existing real label
    already_mask = image > 0
    # assert numpy.all(image[already_mask] == altered[already_mask])
    # Then update our understanding
    return altered


def get_point_subsets(points, cuts):
    '''
    Take in (N, 2) numpy array of pixel points and try to return proper subsets
    where there was a labeled group.
    '''
    for i in range(len(cuts) - 1):
        yield points[cuts[i]:cuts[i+1]]


# What a goddamn mess
def discover_subsets(points):
    '''
    Go through an interactive process to walk through the image and figure out
    by hand which subsets were taken together. Hopefully I'll never have to use
    this again.

    Controls:
        p: toggle the real image on and off
        q, w, e, r, t, y: step the label points by -15, -5, -1, 1, 5, 15 steps
        x: exit
        a: print the current counter
    '''
    cv2.namedWindow("discover")
    sizeNN3 = SIZE + (3,)
    image = numpy.zeros(sizeNN3)
    real = None
    cutpoints = [0]
    counter = max(5, cutpoints[-1])
    while True:
        if real is None:
            cv2.imshow("discover", image.astype(numpy.uint8))
        else:
            cv2.imshow("discover", (0.5*image + 0.5*real).astype(numpy.uint8))
        key = chr(cv2.waitKey(0))
        if key == "p":
            if real is None:
                real = cv2.imread("/home/eric/Downloads/2021-12-01-16-52-30_cam1_1.png")
            else:
                real = None
            continue
        if key == "a":   print(counter)
        elif key == "q": counter -= 15
        elif key == "w": counter -= 5
        elif key == "e": counter -= 1
        elif key == "r": counter += 1
        elif key == "t": counter += 5
        elif key == "y": counter += 15
        elif key == "s":
            cutpoints.append(counter)
            print(f"Cutpoints: {cutpoints}")
        elif key == "x": break
        else:            continue
        image = numpy.zeros(sizeNN3)
        for i in range(max(0, len(cutpoints) - 16), len(cutpoints) - 1):
            image = draw_polygon(image,
                                 (50, 255, 50),
                                 points[cutpoints[i]:cutpoints[i+1]],
                                 (255, 255, 255))
        if cutpoints[-1] < counter - 1:
            image = draw_polygon(image,
                                 (50, 50, 255),
                                 points[cutpoints[-1]:counter],
                                 (255, 255, 255))
    cv2.destroyAllWindows()


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
        choices=["coco-json", "diffgram-json", "hussain-json"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
