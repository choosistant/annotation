from pathlib import Path

import click
import requests
from tqdm import tqdm


class AmazonReviewDataDownloader:
    def __init__(self, url: str, output_dir: str, verbose: bool):
        self._url = url
        self._output_dir = Path(output_dir)
        if self._output_dir.exists() and not self._output_dir.is_dir():
            raise ValueError(f"Output directory {self._output_dir} is not a directory")
        else:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file_path = self._output_dir / self._url.split("/")[-1]
        self._verbose = verbose

    def download(self):
        self._ensure_file_downloaded()
        return self._output_file_path

    def _ensure_file_downloaded(self):
        if self._output_file_path.exists():
            if self._verbose:
                print(f"File {self._output_file_path} already downloaded")
            return
        if self._verbose:
            print(f"Downloading file from {self._url}...")
        with requests.get(self._url, stream=True) as r:
            r.raise_for_status()
            file_size = int(r.headers.get("Content-length", 100000000))
            if self._verbose:
                pbar = tqdm(total=file_size, unit="bytes", unit_scale=True)
            else:
                pbar = None
            with open(self._output_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk))
            if pbar:
                pbar.close()


@click.command(help="Prepares the Amazon review data set.")
@click.option(
    "--url",
    type=click.STRING,
    required=True,
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
    AmazonReviewDataDownloader(**kwargs).download()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
