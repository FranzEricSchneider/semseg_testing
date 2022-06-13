import argparse
import cv2
import logging
import numpy
from pathlib import Path


DIVISOR = 32


def main(base, save):

    # Could be refactored but I don't care
    save.mkdir()
    for dirs in (["img_dir"],
                 ["ann_dir"],
                 ["img_dir", "train"],
                 ["img_dir", "val"],
                 ["ann_dir", "train"],
                 ["ann_dir", "val"]):
        save.joinpath(*dirs).mkdir()

    for i, impath in enumerate(base.glob("*/*/*png")):

        if i % 25 == 0:
            logging.info(f"On iteration {i}")

        image = cv2.imread(str(impath), cv2.IMREAD_UNCHANGED)

        # We want to pad things to be divisible by DIVISOR for Unet
        pad_width = ((0, pad_size(image.shape[0])),
                     (0, pad_size(image.shape[1])))
        if len(image.shape) == 3:
            pad_width += ((0, 0), )

        image = numpy.pad(array=image, pad_width=pad_width)
        savepath = save.joinpath(impath.relative_to(base))
        cv2.imwrite(str(savepath), image)


def pad_size(side):
    if side % DIVISOR == 0:
        return 0
    else:
        return DIVISOR - side % DIVISOR


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
        "-s", "--save-dir",
        help="Directory to create and save the new images in.",
        required=True,
        type=Path,
    )
    args = parser.parse_args()
    assert args.data_path.is_dir()
    assert not args.save_dir.is_dir()
    return args


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()
    main(args.data_path, args.save_dir)
