"""Extract BLIP3o-60k while preserving image/text layout and bytes.

Each source archive is extracted beneath its stem: dalle3.tar -> dalle3/1.jpg,
dalle3/1.txt, etc. Do not re-encode JPEGs or rename samples to hashes: doing so
changes decoded pixels and the sorted index used by DistributedSampler(seed=0).
"""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.JPEG'}


def extract_archive(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError(f'Unsafe archive member: {member.name}')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f'Links/devices are not allowed: {member.name}')
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Archive path escapes output: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f'Unreadable member: {member.name}')
            with stream, target.open('xb') as out:
                shutil.copyfileobj(stream, out, length=8 << 20)


def paired_index(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in IMAGE_SUFFIXES
                  and p.with_suffix('.txt').is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(args.input.glob('*.tar'))
    if not shards:
        raise FileNotFoundError(f'No BLIP3o-60k tar archives in {args.input}')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory; partial extraction is not silently resumed')
    args.output.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        extract_archive(shard, args.output / shard.stem)
        print(f'Extracted original bytes: {shard.name}', flush=True)
    images = paired_index(args.output)
    print(f'Extracted {len(images)} image/text pairs to {args.output}', flush=True)


if __name__ == '__main__':
    main()
