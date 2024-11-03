# This script is a fast way to get a few articles from arxiv in textual form
# Used mostly for prototyping, the bigger dataset is avalible through gsutil
from arxiv2text import arxiv_to_text
from pathlib import Path
import json
import argparse
from tqdm import tqdm


def get_downloader_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, required=True)
    parser.add_argument('-b', '--batch', type=int, default=100)
    parser.add_argument('-m', '--metadata', type=Path, help='Path to metadata', default='metadata.json')
    parser.add_argument('-o', '--output', type=Path, default='arxiv')
    return parser.parse_args()


def load_metadata(path: Path, limit: int, batch: int):
    fp = path.open(encoding='utf-8')
    counter = 0
    while counter * batch < limit:
        counter += 1
        lines = [line for _ in range(batch) if (line := fp.readline()) != '']
        if len(lines) == 0:
            break
        yield [json.loads(line) for line in lines]
    fp.close()


def download(id: str):
    address = f"https://arxiv.org/pdf/{id}"
    filepath = args.output / (id + '.txt')
    if filepath.exists():
        download_stats['skipped'] += 1
        return
    val = arxiv_to_text(address)
    with open(filepath, "w+", encoding='utf-8') as fp:
        fp.write(val)
    download_stats['success'] += 1


if __name__ == '__main__':
    args = get_downloader_args()
    loader = load_metadata(args.metadata, args.count, args.batch)
    args.output.mkdir(exist_ok=True)
    pbar = tqdm(total=args.count)
    download_stats = {'success': 0, 'failed': 0, 'skipped': 0}
    errors = []
    for batch in loader:
        for i, item in enumerate(batch):
            try:
                download(item)
            except Exception as e:
                errors.append(str(e) + '\n')
                download_stats['failed'] += 1
            pbar.update(1)
    print(f"Statistics: {download_stats}")
    with open("errors.txt", "w+") as fp:
        fp.writelines(errors)
