"""Verify, merge and extract the published training/test dataset downloads."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import zipfile


TRAIN_SHA256 = "96523651e5796af4a0f31feec2553e679ce5fca1c07a785e46d85631e5848dc8"
TEST_SHA256 = "ce739afed0068e6c8d3a6a893c13f71d85fefbce18c33dcffe08a109c7867f1e"
TRAIN_PARTS = 10


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(download_dir, output):
    download_dir = Path(download_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    test_zip = download_dir / "test.zip"
    if not test_zip.is_file() or checksum(test_zip) != TEST_SHA256:
        raise ValueError("test.zip is missing or its SHA256 checksum does not match")
    train_zip = download_dir / "train.zip"
    if not train_zip.exists():
        parts = [download_dir / f"train.zip.{index:03d}" for index in range(1, TRAIN_PARTS + 1)]
        missing = [path.name for path in parts if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing training downloads: " + ", ".join(missing))
        partial = download_dir / "train.zip.partial"
        with partial.open("xb") as writer:
            for part in parts:
                with part.open("rb") as reader:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if checksum(partial) != TRAIN_SHA256:
            partial.unlink()
            raise ValueError("Training archive SHA256 mismatch; re-download the corrupted parts")
        os.replace(partial, train_zip)
    elif checksum(train_zip) != TRAIN_SHA256:
        raise ValueError("Existing train.zip does not match the published SHA256")
    # Preflight both archives before extracting anything; never overwrite user files.
    for archive, split in ((train_zip, "train"), (test_zip, "test")):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                target = (output / info.filename).resolve()
                if not target.is_relative_to(output) or not info.filename.startswith(split + "/"):
                    raise ValueError("Unexpected archive path")
                if not info.filename.endswith(".npy") or target.exists():
                    raise FileExistsError(f"Refusing to overwrite or extract an unexpected file: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for archive in (train_zip, test_zip):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output)
        print(f"Extracted {archive.name}")
    print(f"Datasets ready: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", default="../downloads", help="Folder containing test.zip and train.zip.001–010")
    parser.add_argument("--output", default="../data", help="Dataset destination matching the experiment YAMLs")
    args = parser.parse_args()
    prepare(args.download_dir, args.output)


if __name__ == "__main__":
    main()
