'''
Tool to take various label output formats I've been given and turn them into
(H, W) label images with the pixel value being the class.
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
        images = coco_label(image, args.input_file)
    elif args.file_type == "diffgram-json":
        images, vis_image = diffgram_label(image, args.input_file)
    elif args.file_type == "colored-img":
        images = label_by_color(image, args.input_file, args.colors_file)
    elif args.file_type == "H-json":
        images = h_json_label(image, args.input_file)
    else:
        raise NotImplementedError()

    for name, image in images.items():
        if name is None:
            output_path = args.output_path
        else:
            output_path = args.output_path.parent.joinpath(name)

        # Make everything else background
        image[image < 0] = 0
        image = image.astype(numpy.uint8)

        # Fill in known hand-labeled gaps. This is bad, but I tried hard and
        # couldn't find a better way to handle self-intersections. Screw cv2's
        # fillPoly and drawContour.
        image = fill_known_gaps(output_path, image)

        # Save
        cv2.imwrite(str(output_path), image)
        print(f"Saved to {str(output_path)}")

        # Then make a debug version
        if vis_image is None or name is not None:
            vis_image = image
        pyplot.imsave(str(output_path).replace(".png", "_vis.png"), vis_image)


def coco_label(image, json_file):
    '''Process files as they come out of CVAT using the COCO format.'''
    labeldata = json.load(json_file.open("r"))
    annotations = labeldata["annotations"]
    output = {}
    # So there's this annoying happened where the labels from two images got
    # mashed together and written over each other. This "seen" set is used to
    # check that we aren't adding sets of points twice. I don't know if this
    # issue will ever arise again.
    seen = set()
    for image_meta in sorted(labeldata["images"], key=lambda x: x["id"]):
        copied = numpy.ones(image.shape, dtype=image.dtype) * -1
        image_id = image_meta["id"]
        image_name = image_meta["file_name"].split("/")[-1]
        # This is a bit messy, but in the case where the label file only has
        # one image, don't use the image_name to distinguish them.
        if len(labeldata["images"]) == 1:
            image_name = None
        for annotation in annotations:
            if annotation["image_id"] != image_id:
                continue

            segmentation = numpy.array(annotation["segmentation"])
            if str(segmentation) in seen:
                continue
            seen.add(str(segmentation))

            classid = annotation["category_id"]
            copied = draw_polygon(
                copied,
                classid,
                # (x, y) points come out interleaved as [x1, y1, x2, y2, ...]
                # and need to be reshaped into (N, 2)
                segmentation.reshape((-1, 2)),
            )
        output[image_name] = copied
    return output


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
    return {None: image}, vis_image


def h_json_label(image, json_file):
    '''
    Process the json files as they come out of whatever H shared with me first.
    '''
    labeldata = json.load(json_file.open("r"))
    shapes = labeldata["shapes"]
    for shape in shapes:
        assert shape["shape_type"] == "polygon"
        classid = CLASSES[shape["label"].lower()]
        image = draw_polygon(image, classid, shape["points"])
    return {None: image}


def label_by_color(image, ann_img_path, colors_file):

    # Read in the colors
    colors = {}
    with open(colors_file, "r") as infile:
        for line in infile.readlines():
            # This looks complicated, but it just takes
            # 147 203 131 Leaf
            # and turns it into
            # {(147, 203, 131): "leaf"}
            colors[tuple(map(int, line.split()[:3]))] = line.split()[-1].lower()

    # And the annotated image
    ann_img = cv2.cvtColor(cv2.imread(str(ann_img_path)), cv2.COLOR_BGR2RGB)

    # Figure out which parts of the annotated image go with which color/class
    for color, classname in colors.items():
        mask = (ann_img == color).all(axis=2)
        image[mask] = CLASSES[classname]

    return {None: image}


def draw_polygon(image, classid, points, add_points=None):
    points = numpy.array(points)
    altered = image.copy()
    # Add a few decimal points of sub-pixel accuracy, possible with fillPoly
    SUBPIXEL = 4
    subpixel_points = (points * 2**SUBPIXEL).astype(int)
    cv2.fillPoly(img=altered, pts=[subpixel_points], color=classid, shift=SUBPIXEL)

    if add_points is not None:
        # int(x + 0.5) is a trick to round numbers to nearest int
        int_points = (points + 0.5).astype(int)
        altered[int_points.T[1], int_points.T[0]] = add_points
        last = int_points[-1]
        altered[last[1]-2:last[1]+2, last[0]-2:last[0]+2] = add_points

    # This is a little complicated, but basically we want to keep the lowest
    # numbered classes that are not the background. Basically we want to
    # prioritize vines if there is overlap (class 1). This is not 100% right
    # but it's the best choice.
    already_set = image > 0
    unequal = image != altered
    mask = numpy.logical_and(already_set, unequal)
    altered[mask] = numpy.min((altered[mask], image[mask]), axis=0)

    # Then update our understanding
    return altered


# Build up a list of files and the pixel values that we want to floodfill.
# Let's assume for now that we will only do this for vines. Values extracted
# using pinta.
KNOWN_FILL = {
    "2021-12-01-12-36-30_cam1_6.png": [(926, 596), (175, 508), (1555, 912), (1080, 1130), (993, 1236), (880, 1258), (836, 1285), (986, 963), (1128, 1014), (1617, 1920), (1363, 450), (1780, 285)],
    "2021-12-01-13-11-13_cam2_0.png": [(382, 496), (408, 500), (386, 413), (378, 423), (684, 371), (1317, 37), (1305, 234), (1231, 272), (1177, 303), (1605, 908), (1634, 940), (1556, 248), (1969, 986), (1990, 1069), (2183, 163), (1956, 1763), (2055, 338), (1334, 613), (1360, 638), (2157, 1170), (1270, 1101)],
    "2021-12-01-13-31-15_cam3_6.png": [(622, 925), (667, 924), (406, 314), (1721, 993)],
    "2021-12-01-15-16-07_cam1_6.png": [(217, 1580), (345, 1087), (1112, 1200), (1022, 278), (1077, 391), (1150, 438), (1200, 420), (1220, 435), (2092, 1233), (2291, 724), (2319, 732), (2308, 730)],
    "2021-12-01-15-17-19_cam0_1.png": [(863, 265), (763, 1423), (860, 1169), (881, 1147), (889, 1037), (911, 768), (975, 720), (1061, 458), (913, 893), (1186, 112), (1456, 53), (1452, 91), (1812, 350), (994, 198), (2060, 378), (1016, 761), (1845, 800), (1814, 812), (1634, 858), (1597, 869), (927, 960), (1529, 1158), (1740, 1175), (1740, 1185), (1579, 1223), (1746, 1343), (1851, 1653)],
    "2021-12-01-15-18-31_cam2_0.png": [(592, 42), (2250, 54), (1388, 64), (1150, 164), (815, 217), (1271, 328), (274, 522), (361, 537), (1030, 600), (1758, 744), (1744, 786), (1179, 796), (1732, 830), (428, 1102), (477, 1116), (623, 1137), (636, 1161), (661, 1165), (397, 1374), (1073, 1594)],
    "2021-12-01-16-27-46_cam2_3.png": [(511, 146), (1876, 344), (1954, 348), (1900, 350), (1920, 350), (340, 402), (332, 432), (1524, 448), (1734, 518), (2138, 572), (1043, 764), (91, 770), (874, 786), (2215, 848), (1056, 850), (40, 874), (110, 877), (2056, 955), (1070, 968), (1093, 982), (2009, 1032), (440, 1081), (246, 1373)],
    "2021-12-01-16-50-01_cam1_1.png": [(829, 77), (1382, 154), (1854, 160), (712, 162), (608, 178), (227, 292), (869, 338), (659, 408), (1108, 469), (1378, 477), (1353, 504), (991, 518), (1361, 550), (394, 560), (1399, 588), (50, 625), (2196, 660), (933, 729), (912, 799), (913, 1004), (916, 1130), (783, 1236), (803, 1272), (836, 1366)],
    # These two files are the ones blended together in one JSON file, so their
    # output names are currently hard set as what's in the JSON file.
    "2021-12-01-13-11-13_cam2_6 progress pc 1.png": [(61, 194), (99, 223), (1658, 350), (1623, 360), (2068, 526), (2089, 544), (54, 688), (2024, 755), (1906, 770), (1566, 771), (1868, 776), (1747, 778), (1703, 794), (1448, 856), (834, 901), (738, 949), (774, 956), (777, 994), (523, 1022), (1792, 1028), (1773, 1086), (1814, 1090), (1398, 1140), (1374, 1172), (1358, 1193), (945, 1323), (738, 1367), (955, 1382), (678, 1472)],
    "2021-12-01-14-09-33_cam1_0 progress pc 1.png": [(621, 117), (353, 236), (254, 281), (23, 372), (615, 419), (251, 548), (1444, 556), (1453, 582), (797, 604), (333, 609), (1691, 653), (459, 672), (2323, 837), (46, 1026), (1762, 1062), (954, 1269)],
}


def fill_known_gaps(output_file, image):
    if output_file.name in KNOWN_FILL:
        for pixel in KNOWN_FILL[output_file.name]:
            assert image[pixel[1], pixel[0]] == 0, f"Pixel {pixel} != 0..."
            height, width = image.shape
            cv2.floodFill(image,
                          numpy.zeros((height+2, width+2), numpy.uint8),
                          pixel,
                          CLASSES["vine"])
    return image


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
    parser.add_argument(
        "-c", "--colors-file",
        help="F's images come with an associated color file.",
        type=Path,
    )
    parser.add_argument(
        "-i", "--input-file",
        help="Polygonally labelled JSON file or color labeled images, depending"
             " on the file type.",
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
        choices=["coco-json", "diffgram-json", "colored-img", "H-json"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
