"""Command-line interface for synthetic generation and benchmarking."""
import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dicom-edge-bench",
        description="Research-only selective DICOM decoding + edge inference benchmark",
    )
    parser.add_argument("--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Create synthetic DICOM; no patient data")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic_native.dcm"))
    gen.add_argument("--frames", type=int, default=96)
    gen.add_argument("--rows", type=int, default=128)
    gen.add_argument("--columns", type=int, default=128)
    gen.add_argument("--encoding", choices=("native", "jpeg"), default="native")
    gen.add_argument("--seed", type=int, default=347)

    run = subparsers.add_parser("run", help="Compare A/B/C and save JSON + CSV")
    run.add_argument("--dicom", type=Path, required=True)
    run.add_argument("--backend", choices=("highdicom", "pydicom"),
                     default="highdicom")
    run.add_argument("--inference", choices=("torch", "none"), default="torch")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--selection", default="uniform:16")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--prefetch", type=int, default=2)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--torch-threads", type=int, default=1)
    run.add_argument("--out", type=Path, default=Path("results/benchmark.json"))

    args = parser.parse_args(argv)
    if args.command == "generate":
        from .synthetic import create_dicom
        path = create_dicom(args.out, frames=args.frames, rows=args.rows,
                            columns=args.columns, encoding=args.encoding,
                            seed=args.seed)
        print(f"Created synthetic DICOM: {path}")
        return 0
    from .runner import benchmark, save_result
    result = benchmark(
        args.dicom, backend=args.backend, inference_type=args.inference,
        device=args.device, selection=args.selection, batch_size=args.batch_size,
        prefetch=args.prefetch, repeats=args.repeats, warmups=args.warmups,
        torch_threads=args.torch_threads,
    )
    save_result(result, args.out)
    print(json.dumps({
        "summary": result["summary"],
        "device": result["metadata"]["device"],
        "backend": result["metadata"]["backend"],
        "same_pixels": result["metadata"]["identical_decoded_frames_across_all_trials"],
        "results": str(args.out),
    }, indent=2))
    if not result["metadata"]["identical_decoded_frames_across_all_trials"]:
        print("ERROR: Modes returned different decoded-frame hashes")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
