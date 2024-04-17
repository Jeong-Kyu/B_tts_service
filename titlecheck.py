import sys
from ebooklib import epub

def extract_title_from_epub(epub_file):
    book = epub.read_epub(epub_file)

    title = book.get_metadata("DC", "title")
    if title:
        title = title[0][0]
    else:
        title = "Unknown"

    return title

if __name__ == "__main__":
    # Flask 앱으로부터 받은 파일 경로 읽기
    file_path = sys.argv[1].strip()
    new_name = extract_title_from_epub(file_path)
    print(new_name)
