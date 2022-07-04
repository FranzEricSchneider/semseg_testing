'''
The semfire_to_cityscape tool can take in stereo information in a specific
format. This tool allows the automated creation of that stereo data. For now it
is very specific to this particular dataset.
'''

import argparse
import json
from pathlib import Path


# Stereo relationships
CAMERAS = {
    "cam0": "cam1",
    "cam1": "cam0",
    "cam2": "cam3",
    "cam3": "cam2",
}
DISPARITY = {
    "cam0": "cams01",
    "cam1": "cams01",
    "cam2": "cams23",
    "cam3": "cams23",
}


def main(args):
    mapping = {}
    for impath in args.input_dir.glob(f"*{args.filetype}"):

        # Assumes the structure '2021-12-01-15-50-59_cam3_6.png'
        dirname, camera, imname = impath.name.split("_")
        othercam = CAMERAS[camera]
        dispcam = DISPARITY[camera]

        other_impath = args.root_dir.joinpath(dirname,
                                              "stage1_extraction",
                                              "000",
                                              f"rectified_{othercam}_images",
                                              imname)
        disparity = args.root_dir.joinpath(dirname,
                                           "stage1_extraction",
                                           "000",
                                           f"disparities_{dispcam}",
                                           imname.replace(f".{args.filetype}", ".npy"))
        mapping[impath.name] = [str(other_impath), str(disparity)]

    out_path = args.out_dir.joinpath("stereo_filemap.json")
    json.dump(mapping, out_path.open("w"), indent=4)
    print(f"Saved mapping to {out_path.absolute()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "--filetype",
        help="End of the filetype, will be searched for with *<filetype>.",
        default="png",
    )
    parser.add_argument(
        "-i", "--input-dir",
        help="Directory with images to get stereo metadata for.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o", "--out-dir",
        help="Where to save visualized images. If no argument is given, json"
             " file will be stored in the input dir.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-r", "--root-dir",
        help="Directory from which all images are organized inside, according"
             " to the vineyard dataset rules.",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    assert args.input_dir.is_dir()
    assert args.out_dir.is_dir()
    assert args.root_dir.is_dir()

    return args


if __name__ == "__main__":
    main(parse_args())

