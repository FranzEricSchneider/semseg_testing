'''
Tool to visualize labeled images (usually look all black due to class id
being the pixel value).
'''

import argparse
import cv2
from matplotlib import pyplot
from pathlib import Path


def main(args):
    for impath in args.input_dir.glob(f"*{args.filetype}"):
        image = cv2.imread(str(impath), cv2.IMREAD_UNCHANGED)

        # Scale things so classes show up consistently
        image[0, 0] = args.num_classes

        # Decide where to save images
        save_dir = args.out_dir
        if save_dir is None:
            save_dir = args.input_dir
        save_path = save_dir.joinpath(impath.name.replace(
            f".{args.filetype}",
            f"_vis.{args.filetype}",
        ))

        pyplot.imsave(save_path, image, cmap=args.colormap)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--colormap",
        help="Give the string name of a colormap defined here:"
             " https://matplotlib.org/stable/tutorials/colors/colormaps.html.",
        default="viridis",
    )
    parser.add_argument(
        "-f", "--filetype",
        help="End of the filetype, will be searched for with *<filetype>.",
        default="png",
    )
    parser.add_argument(
        "-i", "--input-dir",
        help="Directory with images to visualize.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-n", "--num-classes",
        help="Number of classes (making the max pixel value n-1).",
        default=6,
        type=int,
    )
    parser.add_argument(
        "-o", "--out-dir",
        help="Where to save visualized images. If no argument is given, images"
             " will be stored in the input dir with the _vis suffix.",
        default=None,
        type=Path,
    )
    args = parser.parse_args()

    assert args.input_dir.is_dir()

    return args


if __name__ == "__main__":
    main(parse_args())
