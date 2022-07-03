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
    elif args.file_type == "F-json":
        images = f_json_label(image, args.input_file)
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
        vis_image[0, 0] = len(CLASSES)
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

            segmentation = numpy.array(annotation["segmentation"]).squeeze()
            # Clip this to the first 300 elements because there was a case
            # where the points stopped matching at this point, and 300 points
            # should be plenty enough to determine identity.
            seen_key = str(segmentation[:300])
            if seen_key in seen:
                continue
            seen.add(seen_key)

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
        assert instance["type"] == "polygon", \
               f"Shape type {instance['type']} was not a polygon"
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


def f_json_label(image, json_file):
    '''
    Process the json files as they come out of whatever F shared with me when
    CVAT stopped working.
    '''
    labeldata = json.load(json_file.open("r"))
    instances = labeldata["instances"]
    for instance in instances:
        assert instance["type"] == "polygon"
        classid = CLASSES[instance["className"].lower()]
        points = numpy.array(instance["points"]).squeeze()
        image = draw_polygon(
            image,
            classid,
            # (x, y) points come out interleaved as [x1, y1, x2, y2, ...]
            # and need to be reshaped into (N, 2)
            points.reshape((-1, 2)),
        )
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
    # As a hack, let's prioritize leaves (class 4) over trunk and post
    # (classes 2 and 3)
    altered[altered == 4] = 1.5
    image[image == 4] = 1.5
    altered[mask] = numpy.min((altered[mask], image[mask]), axis=0)
    altered[altered == 1.5] = 4
    image[image == 1.5] = 4

    # Then update our understanding
    return altered


# Build up a list of files and the pixel values that we want to floodfill.
# Let's assume for now that we will only do this for vines. Values extracted
# using pinta.
KNOWN_FILL = {
    # H.S. first set of 10
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
    "2021-12-01-13-11-13_cam2_6 progress pc 1.png": [(61, 194), (803, 208), (821, 218), (99, 223), (1660, 352), (1622, 360), (2068, 526), (2088, 544), (54, 687), (2024, 756), (1904, 769), (1565, 770), (1866, 777), (1741, 778), (1704, 794), (835, 899), (739, 950), (777, 957), (780, 997), (520, 1020), (1795, 1027), (1010, 1032), (1773, 1085), (1811, 1088), (1397, 1138), (1375, 1171), (1358, 1192), (944, 1321), (737, 1366), (955, 1383), (680, 1470)],
    "2021-12-01-14-09-33_cam1_0 progress pc 1.png": [(621, 117), (296, 180), (353, 237), (254, 281), (24, 370), (250, 548), (1443, 556), (1455, 580), (333, 609), (2322, 838), (47, 1027), (662, 1045), (1762, 1062), (798, 1506)],
    # H.S. second set of 10
    "2021-12-01-12-49-31_cam1_5.png": [(2228, 71), (352, 244), (2069, 427), (1492, 448), (93, 478), (65, 484), (1345, 518), (2348, 566), (1381, 588), (245, 589), (1112, 605), (43, 631), (151, 632), (323, 666), (585, 771), (614, 776), (324, 814), (1375, 837), (38, 896), (1570, 993), (1277, 995), (773, 1045), (155, 1061), (1720, 1315), (1368, 1460), (710, 1567)],
    "2021-12-01-13-23-01_cam3_5.png": [(1259, 813)],
    "2021-12-01-13-24-18_cam1_3.png": [(1688, 419), (1496, 435), (1778, 501), (1760, 701), (1760, 727), (1657, 1142), (2313, 1240), (1664, 1281), (1657, 1408)],
    "2021-12-01-13-28-48_cam1_0.png": [(1236, 247), (1731, 1090), (1046, 1610), (1128, 1663), (2169, 1757)],
    "2021-12-01-13-52-38_cam2_6.png": [(1131, 195), (1050, 274), (1062, 312), (1628, 377), (1808, 403), (591, 447), (1336, 624), (1318, 693), (1313, 763), (1330, 802), (1313, 806), (1176, 1071), (1112, 1093), (930, 1149), (1361, 1158), (197, 1280), (1093, 1416), (79, 1430), (2052, 1435), (2076, 1437), (795, 1470), (1630, 1476), (769, 1584), (110, 1611), (129, 1617), (402, 1696), (768, 1698), (548, 1704), (771, 1721), (476, 1722), (806, 1725), (819, 1729), (452, 1779), (822, 1861), (494, 929), (484, 1408)],
    "2021-12-01-14-09-33_cam0_3.png": [(277, 7), (562, 66), (1730, 70), (96, 88), (68, 113), (1092, 121), (2297, 352), (1872, 361), (2248, 421), (2224, 462), (714, 533), (2086, 548), (1684, 553), (1701, 578), (1460, 589), (60, 677), (1745, 708), (27, 764), (1654, 1008), (931, 1037), (100, 1062), (2220, 1149), (2060, 1221)],
    "2021-12-01-14-16-02_cam1_4.png": [(894, 34), (2293, 152), (2280, 177), (1580, 855)],
    "2021-12-01-15-31-55_cam0_2.png": [(2288, 27), (104, 136), (1484, 148), (108, 269), (354, 461), (1777, 531), (1791, 539), (1295, 554), (1790, 556), (1764, 569), (1315, 576), (1328, 646), (974, 790), (1765, 795), (1650, 830), (1764, 904)],
    "2021-12-01-15-46-11_cam3_1.png": [(376, 106), (320, 120), (184, 149), (1652, 779), (856, 1129), (2198, 1661)],
    "2021-12-01-16-18-01_cam1_1.png": [(1475, 24), (1470, 37), (321, 38), (1465, 65), (803, 87), (636, 191), (2363, 198), (660, 244), (1634, 347), (1582, 419), (1719, 515), (122, 626), (671, 725), (320, 742), (674, 747), (700, 747), (2057, 856), (2112, 908), (2190, 974), (2288, 1041), (2076, 1184), (2060, 1263), (2056, 1278)],
    # H.S. fifth set of 10
    "2021-12-01-15-37-26_cam3_5.png": [(1276, 54), (1625, 159), (1836, 208), (1525, 212), (1871, 217), (1794, 235), (1813, 248), (1370, 268), (1924, 549), (2294, 588), (1139, 593), (1136, 638), (1132, 639)],
    "2021-12-01-13-24-18_cam2_0.png": [(2090, 207), (1114, 279), (599, 412), (696, 424), (516, 554), (1284, 563)],
    "2021-12-01-15-18-31_cam1_4.png": [(2413, 68), (1531, 73), (1502, 81), (1612, 84), (1601, 136), (1705, 144), (1785, 156), (1663, 214), (1349, 296)],
    "2021-12-01-16-10-45_cam2_0.png": [(704, 116), (2354, 116), (828, 153), (656, 160), (1019, 291), (2276, 367), (1564, 419), (1540, 449), (1528, 466), (284, 494), (1656, 498), (2167, 534), (2249, 538), (2115, 540), (2131, 540), (1073, 545), (886, 578), (965, 618), (1848, 653), (2057, 763), (2051, 797), (2064, 814), (2051, 818), (783, 819), (2068, 822), (2052, 826), (779, 827), (1097, 841), (2086, 858), (2043, 933), (1106, 972), (2151, 986), (2083, 987), (2114, 1017), (1130, 1029), (2093, 1033), (1133, 1039), (2033, 1079), (2334, 1098), (790, 1123), (310, 1167), (328, 1200), (2258, 1270), (845, 1361), (2078, 1746)],
    "2021-12-01-14-22-41_cam2_0.png": [(1980, 77), (1194, 88), (742, 102), (1580, 102), (1129, 112), (1108, 134), (1121, 138), (72, 145), (1567, 253), (516, 256), (594, 270), (331, 271), (356, 272), (621, 278), (784, 278), (535, 286), (1507, 286), (103, 316), (532, 322), (338, 323), (373, 331), (128, 335), (73, 356), (475, 356), (526, 359), (1954, 450), (520, 452), (499, 465), (519, 468), (589, 484), (802, 528), (1781, 590), (528, 624), (2320, 664), (1049, 688), (1048, 726), (2401, 749), (1255, 870)],
    "2021-12-01-13-47-55_cam1_1.png": [(1989, 5), (534, 14), (590, 14), (632, 37), (2282, 53), (980, 63), (345, 67), (175, 88), (422, 91), (440, 100), (426, 105), (841, 105), (807, 108), (903, 167), (915, 175), (1901, 266), (1856, 270), (360, 522), (207, 699), (2381, 1255), (2050, 1310), (547, 1568)],
    "2021-12-01-14-10-55_cam0_5.png": [(1789, 106), (1775, 118), (1298, 132), (1314, 137), (509, 217), (485, 218), (1558, 221), (1741, 221), (436, 223), (486, 236), (445, 272), (2099, 274), (113, 279), (2222, 286), (2307, 298), (2322, 299), (143, 324), (58, 331), (534, 360), (321, 376), (530, 401), (501, 407), (534, 425), (238, 472), (230, 506), (1474, 507), (369, 517), (624, 640)],
    "2021-12-01-13-37-09_cam3_4.png": [(2353, 231), (56, 416), (2153, 494), (2081, 547), (367, 588), (1566, 626), (1584, 630), (1599, 632), (1964, 658), (2073, 671), (1542, 709), (1554, 731), (1522, 751), (1580, 760), (2049, 761), (1538, 762), (1853, 763), (1520, 765), (1764, 826), (1883, 826), (1912, 901), (1651, 911), (2246, 964), (1512, 994), (1523, 1024), (1392, 1114), (2093, 1117), (2096, 1132), (2286, 1141), (1412, 1159), (2123, 1201), (2068, 1383), (1890, 1384), (1553, 1639)],
    "2021-12-01-12-35-00_cam1_5.png": [(1042, 1068), (277, 1393), (2350, 1629)],
    "2021-12-01-16-44-49_cam3_2.png": [(1782, 24), (1757, 210), (1705, 447), (1721, 454), (1799, 492), (1782, 518), (2329, 615), (1777, 619), (2277, 623), (2199, 642), (2253, 644), (1788, 648), (2268, 668), (1865, 671), (2173, 681), (2312, 706), (497, 819), (517, 819), (748, 824), (581, 865), (441, 968), (1914, 1020), (2009, 1078), (1577, 1244), (561, 1477)],
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
        help="Furqan's images come with an associated color file.",
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
        choices=["coco-json", "diffgram-json", "colored-img", "H-json", "F-json"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
