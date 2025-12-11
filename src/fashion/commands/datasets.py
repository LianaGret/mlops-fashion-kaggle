from __future__ import annotations

from pathlib import Path
from typing import Literal

from fashion.console import (
    console,
    create_progress,
    create_table,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_success,
    print_warning,
)

DatasetName = Literal["articles", "customers", "transactions", "images"]

S3_BUCKET = "guzbkm-mlops-project"
S3_PREFIX = "fashion-dataset"

AVAILABLE_DATASETS: dict[DatasetName, dict[str, str]] = {
    "articles": {
        "description": "Article metadata (product info, colors, categories)",
        "local_path": "data/raw/articles.csv",
        "s3_key": "articles.csv",
    },
    "customers": {
        "description": "Customer metadata (demographics, club membership)",
        "local_path": "data/raw/customers.csv",
        "s3_key": "customers.csv",
    },
    "transactions": {
        "description": "Transaction history (purchases)",
        "local_path": "data/raw/transactions_train.csv",
        "s3_key": "transactions_train.csv",
    },
    "images": {
        "description": "Product images",
        "local_path": "data/raw/images/",
        "s3_key": "images/",
    },
}


class Datasets:
    """Dataset management commands.

    Commands:
        download    Download datasets from remote storage
        list        List available datasets
        info        Show dataset information
    """

    def download(
        self,
        *names: DatasetName,
        all: bool = False,
        force: bool = False,
    ) -> None:
        """Download datasets from S3 via DVC import-url.

        Args:
            names: Dataset names to download (articles, customers, transactions, images)
            all: Download all available datasets
            force: Force re-download even if files exist
        """
        import subprocess

        print_header("Dataset Download", "Fetching data from S3 via DVC")

        if all:
            targets = list(AVAILABLE_DATASETS.keys())
        elif names:
            targets = list(names)
        else:
            print_warning("No datasets specified. Use --all or provide dataset names.")
            print_info("Available datasets: " + ", ".join(AVAILABLE_DATASETS.keys()))
            return

        invalid = [name for name in targets if name not in AVAILABLE_DATASETS]
        if invalid:
            print_error(f"Unknown datasets: {', '.join(invalid)}")
            print_info("Available: " + ", ".join(AVAILABLE_DATASETS.keys()))
            return

        print_info(f"Downloading {len(targets)} dataset(s)...")
        console.print()

        with create_progress() as progress:
            task = progress.add_task("Downloading datasets", total=len(targets))

            for dataset_name in targets:
                dataset_info = AVAILABLE_DATASETS[dataset_name]
                local_path = dataset_info["local_path"]
                s3_key = dataset_info["s3_key"]
                s3_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/{s3_key}"

                progress.update(task, description=f"Downloading {dataset_name}")

                target_path = Path(local_path)
                dvc_file = Path(f"{local_path}.dvc")

                if target_path.exists() and not force:
                    progress.console.print(
                        f"  [muted]Skipping {dataset_name} (exists, use --force)[/muted]"
                    )
                    progress.advance(task)
                    continue

                try:
                    if force and dvc_file.exists():
                        dvc_file.unlink()
                        subprocess.run(
                            ["git", "rm", "--cached", str(dvc_file)],
                            capture_output=True,
                        )

                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    result = subprocess.run(
                        ["dvc", "import-url", s3_url, str(local_path)],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        progress.console.print(f"  [success]>[/success] Downloaded {dataset_name}")
                    else:
                        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
                except Exception as error:
                    progress.console.print(f"  [error]x[/error] Failed {dataset_name}: {error}")

                progress.advance(task)

        console.print()
        print_success("Download complete!")

    def list(self) -> None:
        """List all available datasets."""
        print_header("Available Datasets")
        console.print()

        table = create_table("Datasets", ["Name", "Description", "Path"])
        for name, info in AVAILABLE_DATASETS.items():
            path = Path(info["local_path"])
            status = "[green]local[/green]" if path.exists() else "[muted]remote[/muted]"
            table.add_row(name, info["description"], f"{info['local_path']} ({status})")

        console.print(table)

    def info(self, name: DatasetName) -> None:
        """Show detailed information about a dataset.

        Args:
            name: Dataset name
        """
        if name not in AVAILABLE_DATASETS:
            print_error(f"Unknown dataset: {name}")
            return

        info = AVAILABLE_DATASETS[name]
        path = Path(info["local_path"])
        s3_url = f"s3://{S3_BUCKET}/{S3_PREFIX}/{info['s3_key']}"

        print_header(f"Dataset: {name}")
        console.print()

        print_key_value("Description", info["description"])
        print_key_value("Local Path", info["local_path"])
        print_key_value("S3 URL", s3_url)
        print_key_value("Local Status", "Available" if path.exists() else "Not downloaded")

        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print_key_value("Size", f"{size_mb:.2f} MB")

                if path.suffix == ".csv":
                    import pandas as pd

                    df = pd.read_csv(path, nrows=5)
                    print_key_value("Columns", str(len(df.columns)))
                    console.print()
                    console.print("[muted]Preview (first 5 rows):[/muted]")
                    console.print(df.to_string())
            elif path.is_dir():
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                print_key_value("Files", str(file_count))
