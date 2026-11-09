"""按 Notebook 023 的固定协议复现三模型投票。"""
import argparse
import hashlib
import json
from pathlib import Path

from sentiment_analysis.paths import REPORTS_DIR
from sentiment_analysis.voting import VOTING_PROTOCOL_PATH, evaluate_fixed_voting, save_voting_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "reproduced",
                        help="Directory for the reproduced CSVs (default: reports/reproduced).")
    args = parser.parse_args()

    protocol_bytes = VOTING_PROTOCOL_PATH.read_bytes()
    print(f"Protocol SHA-256: {hashlib.sha256(protocol_bytes).hexdigest()}", flush=True)
    comparison, predictions = evaluate_fixed_voting(json.loads(protocol_bytes))

    save_voting_results(args.output_dir, comparison, predictions)
    print(f"Aligned test samples: {len(predictions):,}")
    print(comparison.round(4).to_string(index=False))
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
