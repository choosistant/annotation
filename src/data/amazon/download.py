import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import click
import requests
from tqdm import tqdm


@dataclass
class CategoryItem:
    name: str
    reviews_url: str
    reviews_count: int
    ratings_url: str
    ratings_count: int

    @property
    def slug(self) -> str:
        return self.name.replace(" ", "-").lower()


class AmazonReviewDataDownloader:
    def __init__(
        self, meta_data_path: str, categories: str, output_dir: str, verbose: bool
    ):
        self._meta_data_path = Path(meta_data_path)
        if not self._meta_data_path.exists():
            raise ValueError(f"Meta data file {self._meta_data_path} does not exist")

        with open(self._meta_data_path, "r") as f:
            self._meta_data = json.load(f)

        self._available_categories = [
            CategoryItem(**item) for item in self._meta_data["categories"]
        ]

        self._selected_categories = self._parse_categories(
            input_categories=categories,
            available_categories=self._available_categories,
        )

        self._output_dir = Path(output_dir)
        if self._output_dir.exists() and not self._output_dir.is_dir():
            raise ValueError(f"Output directory {self._output_dir} is not a directory")
        else:
            self._output_dir.mkdir(parents=True, exist_ok=True)

        self._verbose = verbose

    def run(self):
        for category in self._selected_categories:
            file_name = f"reviews-{category.slug}-{category.reviews_count}.json.gz"
            output_file_path = self._output_dir / file_name
            self._download_file(
                url=category.reviews_url, output_file_path=output_file_path
            )

    def _download_file(self, url: str, output_file_path: Path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            file_size = int(r.headers.get("Content-length", 100000000))
            # Check whether there exist a file with matching name and size.
            if (
                output_file_path.exists()
                and file_size == output_file_path.stat().st_size
            ):
                if self._verbose:
                    print(
                        f"Skipping file {output_file_path} as it is already downloaded."
                    )
                return
            else:
                if self._verbose:
                    print(f"Downloading file from {url}...")
            pbar = None
            if self._verbose:
                pbar = tqdm(total=file_size, unit="bytes", unit_scale=True)
            with open(output_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk))
            if pbar:
                pbar.close()

    def _parse_categories(
        self, input_categories: str, available_categories: List[CategoryItem]
    ) -> List[CategoryItem]:
        selected_categories: List[CategoryItem] = []
        if (
            input_categories is None
            or input_categories == ""
            or input_categories.lower() == "all"
        ):
            selected_categories = [item for item in available_categories]
        else:
            selected_categories = []
            avail_catagory_names = [item.name.lower() for item in available_categories]
            for c in input_categories.split(","):
                try:
                    category_idx = avail_catagory_names.index(c.lower().strip())
                except ValueError:
                    raise ValueError(
                        f"Name '{c}' is not a valid category. Cannot be found in meta data."
                    )
                selected_categories.append(available_categories[category_idx])
        return selected_categories


@click.command(help="Prepares the Amazon review data set.")
@click.option(
    "--meta-data-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "--categories",
    type=click.STRING,
    required=False,
    default="all",
    show_default=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "--verbose",
    is_flag=True,
    show_default=True,
    default=False,
)
def main(**kwargs):
    AmazonReviewDataDownloader(**kwargs).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
