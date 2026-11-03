import argparse
import os

from config import settings
from vectorstore import extract_pages, ingest_pages


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a local .txt, .md or .pdf file into the FAISS vector store."
    )
    parser.add_argument("--file", required=True, help="Path to the file to ingest.")
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name (sub-folder under faiss_store/). Default: 'default'",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Max characters per chunk. Default: {settings.chunk_size}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Character overlap between consecutive chunks. Default: {settings.chunk_overlap}",
    )
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        data = f.read()
    name = os.path.basename(args.file)
    pages = extract_pages(name, data)

    print(f"Loaded '{args.file}' ({len(data):,} bytes, {len(pages)} page(s))")
    print(f"Ingesting into collection '{args.collection}' ...")

    added, total = ingest_pages(
        pages,
        source=name,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Done. {added} chunks added; collection now holds {total}.")


if __name__ == "__main__":
    main()
