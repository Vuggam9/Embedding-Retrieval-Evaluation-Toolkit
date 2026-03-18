from pathlib import Path

import pandas as pd



def save_reports(results_df: pd.DataFrame, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "benchmark_results.csv"
    md_path = reports_dir / "benchmark_summary.md"

    results_df.to_csv(csv_path, index=False)

    sorted_df = results_df.sort_values(by=["mrr", "recall_at_k", "hit_rate"], ascending=False)
    lines = [
        "# Benchmark Summary",
        "",
        "## Top Configurations",
        "",
        sorted_df.head(5).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "Higher MRR indicates the correct document was retrieved closer to rank 1.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
