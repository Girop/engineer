from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pathlib import Path
import json
import io
import argparse
from dataclasses import dataclass
from tqdm import tqdm
import nltk
nltk.download('words')
from nltk.corpus import words


def get_pdf2text_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=Path,
        required=True,
        help="Directory containing PDFs with arxiv versioning system in names or single PDF"
    )
    parser.add_argument('-o', '--output', type=Path, required=True, help="Root dir for result")
    return parser.parse_args()


@dataclass
class Converted:
    content: str
    path: Path


def path2text(path: Path) -> Converted:
    with open(path, 'rb') as fp:
        pdf_file = io.BytesIO(fp.read())
    resource_manager = PDFResourceManager()
    text_stream = io.StringIO()
    laparams = LAParams()

    device = TextConverter(resource_manager, text_stream, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)
    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)

    extracted_text = text_stream.getvalue()
    text_stream.close()
    return Converted(extracted_text, path)


ENGLISH_WORDS = set(words.words())

def verify_content(content: str, threshold: float) -> bool:
    content_words = [word.strip().lower() for word in content.split()]
    english_word_count = sum(1 for word in content_words if word in ENGLISH_WORDS)

    if len(content) == 0:
        return False

    proportion_english = english_word_count / len(content_words)
    return proportion_english >= threshold


def save_conversions(converted: Converted):
    final_path: Path = out / converted.path.with_suffix('.txt').name
    with open(final_path, "w+", encoding='utf-8') as fp:
        fp.write(converted.content)


def run(path: Path, stats: dict[str, int]) -> list[Converted]:
    path_arr = list(path.iterdir()) if path.is_dir() else [path]

    result = []
    print("Converting to text")
    for file in tqdm(path_arr):
        final_path: Path = out / file.with_suffix('.txt').name
        if final_path.exists():
            print(f"Skipping {final_path}")
            stats['skipped'] += 1
            continue
        try:
            converted = path2text(file)
            if verify_content(converted.content, 0.20):
                save_conversions(converted)
                stats['successful'] += 1
            else:
                stats['unreadable'] += 1
        except Exception as e:
            stats['failed'] += 1
            print(f"Failed conversion: {e}")
    return result


if __name__ == '__main__':
    args = get_pdf2text_args()
    input_: Path = args.input
    out: Path = args.output

    conversion_stats = {
         'failed': 0,
         'unreadable': 0,
         'successful': 0,
         'skipped': 0,
     }

    stat_save_path = Path(f"stats/{input_.name}.json")
    out.mkdir(exist_ok=True)
    stat_save_path.parent.mkdir(exist_ok=True)
    converted_arr = run(input_, conversion_stats)
    with open(stat_save_path, "w+") as fp:
        json.dump(conversion_stats, fp)
