import argparse
import cv2
import numpy
from pathlib import Path

'''
When run, creates
    1) a set of fake images that are (sort of) like the vine images
    2) a corresponding set of label images
The goal is that these can be used to test the basic functioning of the
trainable networks.
'''


class Color:
    '''Generate RGB color samples.'''
    def __init__(self, mean, stddev):
        assert len(mean) == 3
        assert len(stddev) == 3
        self.mean = numpy.array(mean)
        self.stddev = numpy.array(stddev)

    def fill(self, num_samples):
        '''Generate an Nx3 vector of this color, within 0-255.'''
        vector = numpy.random.normal(
            loc=self.mean,
            scale=self.stddev,
            size=(num_samples, 3),
        )
        # Let's bound the bottom by mirroring but just clip the top, because I
        # think there will be more action near the bottom and it's harder to
        # mirror around 255.
        vector[vector < 0] *= -1
        vector = numpy.clip(vector, 0, 255)
        # Return in the right type
        return vector.astype(numpy.uint8)


def background(image, label, classid):
    mask = numpy.all(image == 0, axis=2)
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))
    label[mask] = classid


def vine(image, label, classid):
    # Rough rule of thumb for the number of vines
    number = int(image.shape[1] / 20)
    # Generate starting/ending points for vines naively
    rows = numpy.random.randint(0, image.shape[0], size=(number, 2))
    cols = numpy.random.randint(0, image.shape[1], size=(number, 2))
    # Color in the labels first
    for row, col in zip(rows, cols):
        cv2.line(
            img=label,
            pt1=(col[0], row[0]),
            pt2=(col[1], row[1]),
            color=classid,
            thickness=numpy.random.randint(6, 16),
        )
    # Then color the fake image in using those labels as a mask
    mask = label == classid
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))


def post(image, label, classid):
    # Short-circuit randomly
    if numpy.random.random() < 0.05:
        return

    # Generate post near the top and bottom, vertical
    col = numpy.random.randint(0, image.shape[0])
    offset = numpy.random.randint(0, int(image.shape[1] / 10))
    # Color in the label first
    cv2.line(
        img=label,
        pt1=(col, offset),
        pt2=(col, image.shape[0]-offset-1),
        color=classid,
        thickness=numpy.random.randint(35, 45),
    )
    # Then color the fake image in using those labels as a mask
    mask = label == classid
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))


def leaves(image, label, classid):
    # Rough rule of thumb for the number of vines
    number = numpy.random.randint(0, 3)
    # Generate starting/ending points for vines naively
    rows = numpy.random.randint(0, image.shape[0], size=number)
    cols = numpy.random.randint(0, image.shape[1], size=number)
    # Color in the labels first
    for row, col in zip(rows, cols):
        cv2.circle(
            img=label,
            center=(col, row),
            radius=numpy.random.randint(30, 60),
            color=classid,
            thickness=-1,
        )
    # Then color the fake image in using those labels as a mask
    mask = label == classid
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))


def trunk(image, label, classid):
    # Generate trunk horizontally
    row = numpy.random.randint(0, int(image.shape[0] / 2))
    offset = numpy.random.randint(int(image.shape[1] / 10),
                                  int(image.shape[1] / 3))
    # Color in the label first
    cv2.line(
        img=label,
        pt1=(offset, row),
        pt2=(image.shape[1]-offset-1, row),
        color=classid,
        thickness=numpy.random.randint(15, 25),
    )
    # Then color the fake image in using those labels as a mask
    mask = label == classid
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))


def sign(image, label, classid):
    # Short-circuit randomly
    if numpy.random.random() < 0.6:
        return

    # Generate post near the top and bottom, vertical
    col = numpy.random.randint(0, image.shape[0])
    offset = numpy.random.randint(int(image.shape[1] / 5),
                                  int(image.shape[1] / 3))
    # Color in the label first
    cv2.line(
        img=label,
        pt1=(col, offset),
        pt2=(col, image.shape[0]-offset-1),
        color=classid,
        thickness=numpy.random.randint(45, 65),
    )
    # Then color the fake image in using those labels as a mask
    mask = label == classid
    image[mask] = COLORS[classid].fill(numpy.count_nonzero(mask))


SIZE = (2048, 2448)
ARTISTS = {
    0: background,
    1: vine,
    2: post,
    3: leaves,
    4: trunk,
    5: sign,
}
COLORS = {
    # Background
    0: Color([0, 0, 0], [20, 20, 20]),
    # Vine
    1: Color([190, 170, 120], [40, 20, 20]),
    # Post
    2: Color([25, 60, 25], [10, 30, 10]),
    # Leaves
    3: Color([220, 210, 160], [10, 10, 10]),
    # Trunk
    4: Color([120, 130, 80], [40, 40, 20]),
    # Sign
    5: Color([250, 250, 250], [15, 15, 15]),
}


def main(img_dir, lbl_dir, number):
    for i, (image, label) in enumerate(generate(number)):
        cv2.imwrite(
            str(img_dir.joinpath(f"{i:06}.png")),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(lbl_dir.joinpath(f"{i:06}.png")),
            label,
        )


def generate(number):
    for _ in range(number):
        image = numpy.zeros(SIZE + (3,), dtype=numpy.uint8)
        label = numpy.zeros(SIZE, dtype=numpy.uint8)
        for classid, artist in ARTISTS.items():
            artist(image, label, classid)
        yield image, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path through obstacles")
    parser.add_argument(
        "-i", "--outimgs",
        help="Path to a directory where the generated images go.",
        type=Path,
    )
    parser.add_argument(
        "-l", "--outlbls",
        help="Path to a directory where the generated label images go. This"
             " should not be the same as outimgs because the generated images"
             " will share the same name.",
        type=Path,
    )
    parser.add_argument(
        "-n", "--number",
        help="Number of fakes to generate.",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    assert args.outimgs.is_dir()
    assert args.outlbls.is_dir()

    main(img_dir=args.outimgs,
         lbl_dir=args.outlbls,
         number=args.number)
