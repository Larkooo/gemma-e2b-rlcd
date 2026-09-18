"""Create synthetic integration fixtures without using personal media."""

import argparse
import subprocess
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), (255, 0, 0)).save(root / "red.png")
    subprocess.run(
        [
            "say",
            "-v",
            "Samantha",
            "-o",
            str(root / "dog.wav"),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            "A dog is sleeping on the sofa.",
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=224x224:r=2:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=224x224:r=2:d=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(root / "red-blue.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(root / "red-blue.mp4"),
            "-i",
            str(root / "dog.wav"),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            "apad",
            "-t",
            "4",
            str(root / "red-blue-speech.mp4"),
        ],
        check=True,
    )
    print(root)


if __name__ == "__main__":
    main()
