import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="karaoke")
    sub = parser.add_subparsers(dest="command", required=True)

    sep = sub.add_parser("separate", help="split songs into karaoke stems")
    sep.add_argument("songs", nargs="+", type=Path)
    sep.add_argument("-o", "--out", type=Path, default=Path("output"), help="one subfolder per song is created here")
    sep.add_argument("--overwrite", action="store_true", help="re-run even if stems already exist")

    sing = sub.add_parser("sing", help="open the karaoke window")
    sing.add_argument("song", nargs="?", type=Path)

    args = parser.parse_args()

    if args.command == "sing":
        from karaoke.app import run

        raise SystemExit(run(args.song))

    from karaoke.separation import StemSeparator

    separator = StemSeparator()
    for song in args.songs:
        start = time.perf_counter()
        stems = separator.separate(song, args.out / song.stem, overwrite=args.overwrite)
        print(f"{song.name}: {time.perf_counter() - start:.1f}s (stem gain {stems.gain:.3f})")
        for name, path in vars(stems).items():
            if isinstance(path, Path):
                print(f"  {name:15} {path}")
