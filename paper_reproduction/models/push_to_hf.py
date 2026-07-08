"""Publish the exported model folder to the Hugging Face Hub.

Uploads the folder produced by export_hemato_probe.py (the skops probe, config,
and model card) to a Hub repo. You run this; it needs an HF token with write
access (huggingface-cli login, or --token).

Usage:
  python models/push_to_hf.py --repo-id NIATS-UFU/parkinson-spiral-dinov2-hemato
  python models/push_to_hf.py --repo-id <org/name> --model-dir models/exported --private
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. NIATS-UFU/<name>")
    ap.add_argument("--model-dir", type=Path, default=REPO / "models" / "exported")
    ap.add_argument("--token", default=None, help="HF token (else uses the cached login)")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError as err:
        raise SystemExit("huggingface_hub is required: pip install -e '.[hf]'") from err

    if not (args.model_dir / "parkinson_spiral_dinov2_hemato.skops").is_file():
        raise SystemExit(f"no exported model in {args.model_dir}. Run models/export_hemato_probe.py first.")
    if not (args.model_dir / "README.md").is_file():
        print("WARNING: no README.md (model card) in the export folder — the Hub page will be bare.", file=sys.stderr)

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.model_dir),
        token=args.token,
        commit_message="Add DINOv2+Hemato spiral PD probe (paper release)",
    )
    print(f"pushed {args.model_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
