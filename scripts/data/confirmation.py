'''
Tool to take visualized annotations and overlay them over the original images
for confirmation of quality. WARNING - gif creation is very slow at the moment.
'''

import argparse
import cv2
import numpy
from pathlib import Path
import subprocess
import tempfile


def main(anndir, imgdir, out):
    for annpath in sorted(anndir.glob("*png")):
        impath = imgdir.joinpath(annpath.name)
        assert impath.is_file()
        save_gif(cv2.imread(str(impath)),
                 cv2.imread(str(annpath)),
                 out,
                 impath.name)
        print(f"Saved {impath.name} as gif")


def save_gif(img, ann, out, name):
    mixed = [
        ((img * ratio) + (ann * (1 - ratio))).astype(numpy.uint8)
        for ratio in (0.75, 0.5, 0.25)
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        cv2.imwrite(f"{temp_dir}/1.png", img)
        for i, miximg in enumerate(mixed):
            cv2.imwrite(f"{temp_dir}/{2+i}.png", miximg)
        cv2.imwrite(f"{temp_dir}/{len(mixed)+2}.png", ann)
        for i, miximg in enumerate(reversed(mixed)):
            cv2.imwrite(f"{temp_dir}/{len(mixed)+3+i}.png", miximg)
        command = [
            "convert",
            "-delay", "60",
            "-loop", "0",
            f"{temp_dir}/*.png",
            str(out.absolute()) + "/" + name.replace(".png", ".gif"),
        ]
        subprocess.call(command)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-a", "--annotations-dir",
        help="Directory with visualized annotations, should match image names.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-i", "--img-dir",
        help="Directory with image, should match annotation names.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory where the output should go (gifs).",
        required=True,
        type=Path,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for directory in (args.annotations_dir, args.img_dir, args.output_dir):
        assert directory.is_dir(), f"{directory} was not findable"
    main(anndir=args.annotations_dir,
         imgdir=args.img_dir,
         out=args.output_dir)
