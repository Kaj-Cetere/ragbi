import argparse
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from supabase import Client, create_client

from config_commentaries import MAIN_COMMENTATORS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
SEFARIA_TEXT_API_TEMPLATE = "https://www.sefaria.org/api/texts/{ref}?commentary=1&context=0"


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_ROOT_CANDIDATES = [
    SCRIPT_DIR / "Shulchan-Arukh-Extracted",
    Path.cwd() / "Shulchan-Arukh-Extracted",
    Path.home() / "Shulchan-Arukh-Extracted",
    Path.home() / "Documents" / "Shulchan-Arukh-Extracted",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_text(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw)
    text = html.unescape(text)
    text = re.sub(r"<i data-commentator[^>]+></i>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def flatten_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(flatten_strings(item))
        return out
    return []


def section_from_book(base_book: str) -> str:
    if "," in base_book:
        return base_book.split(",", 1)[1].strip()
    return base_book.strip()


def allowed_commentators(base_book: str) -> List[str]:
    for prefix, commentators in MAIN_COMMENTATORS.items():
        if base_book.startswith(prefix):
            return commentators
    return []


def resolve_base_text_path(local_root: Path, base_book: str) -> Optional[Path]:
    return (
        local_root
        / "Halakhah"
        / "Shulchan Arukh"
        / base_book
        / "Hebrew"
        / "merged.json"
    )


def get_supabase_client() -> Client:
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
    return create_client(supabase_url, supabase_key)


def page_base_chunks(
    supabase: Client,
    page_size: int,
    offset: int,
    book_prefix: Optional[str],
) -> List[Dict[str, Any]]:
    query = (
        supabase.table("sefaria_text_chunks")
        .select("sefaria_ref,metadata")
        .order("chunk_index")
    )
    if book_prefix:
        query = query.contains("metadata", {"book": book_prefix})
    response = query.range(offset, offset + page_size - 1).execute()
    return response.data or []


def discover_merged_files(local_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in local_root.rglob("merged.json"):
        if not path.is_file():
            continue
        path_text = path.as_posix()
        if "/cltk-full/" in path_text or "/cltk-flat/" in path_text:
            continue
        if "/Hebrew/" not in path_text:
            continue
        files.append(path)
    return files


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Failed loading %s: %s", path, exc)
        return None


def build_source_index(local_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    files = discover_merged_files(local_root)
    logger.info("Discovered %d merged.json files under %s", len(files), local_root)

    required: List[Tuple[str, str]] = []
    for base_book, commentators in MAIN_COMMENTATORS.items():
        section = section_from_book(base_book)
        for commentator in commentators:
            required.append((commentator, section))

    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for commentator, section in required:
        expected = resolve_expected_commentary_path(local_root, commentator, section)
        if expected and expected.exists():
            data = load_json(expected)
            if data:
                index[(commentator, section)] = {"path": expected, "title": str(data.get("title", "")), "data": data}
                logger.info("Mapped (%s | %s) -> %s [deterministic]", commentator, section, expected)
                continue

        best_score = -1
        best_entry: Optional[Dict[str, Any]] = None

        norm_commentator = normalize(commentator)
        norm_section = normalize(section)

        for path in files:
            data = load_json(path)
            if not data:
                continue

            title = str(data.get("title", ""))
            candidate_text = f"{path.as_posix()} {title}"
            norm_candidate = normalize(candidate_text)

            score = 0
            if norm_commentator in norm_candidate:
                score += 10
            if norm_section in norm_candidate:
                score += 5
            if normalize("Shulchan Arukh") in norm_candidate:
                score += 2
            if normalize("Commentary") in norm_candidate:
                score += 1

            if score > best_score:
                best_score = score
                best_entry = {"path": path, "title": title, "data": data}

        if best_entry and best_score >= 10:
            index[(commentator, section)] = best_entry
            logger.info(
                "Mapped (%s | %s) -> %s",
                commentator,
                section,
                best_entry["path"],
            )
        else:
            logger.warning("No local source file found for (%s | %s)", commentator, section)

    return index


def resolve_expected_commentary_path(local_root: Path, commentator: str, section: str) -> Optional[Path]:
    if commentator == "Mishnah Berurah" and section == "Orach Chayim":
        return (
            local_root
            / "Halakhah"
            / "Shulchan Arukh"
            / "Commentary"
            / "Mishnah Berurah"
            / "Mishnah Berurah"
            / "Hebrew"
            / "merged.json"
        )

    if commentator == "Ba'er Hetev":
        return (
            local_root
            / "Halakhah"
            / "Shulchan Arukh"
            / "Commentary"
            / "Ba'er Hetev"
            / f"Ba'er Hetev on Shulchan Arukh, {section}"
            / "Hebrew"
            / "merged.json"
        )

    return None


def build_base_text_index(local_root: Path) -> Dict[str, Dict[str, Any]]:
    base_books = [book for book in MAIN_COMMENTATORS if book.startswith("Shulchan Arukh, ")]
    index: Dict[str, Dict[str, Any]] = {}

    for base_book in base_books:
        path = resolve_base_text_path(local_root, base_book)
        if not path or not path.exists():
            logger.warning("No local base text found for %s", base_book)
            continue
        data = load_json(path)
        if not data:
            continue
        index[base_book] = {"path": path, "title": str(data.get("title", "")), "data": data}
        logger.info("Mapped base text %s -> %s", base_book, path)

    return index


def get_value_at_siman_seif(
    text_obj: Any,
    siman: int,
    seif: int,
    subsection: Optional[str] = None,
) -> Any:
    if isinstance(text_obj, list):
        if siman - 1 < 0 or siman - 1 >= len(text_obj):
            return None
        siman_val = text_obj[siman - 1]
        if isinstance(siman_val, list):
            if seif - 1 < 0 or seif - 1 >= len(siman_val):
                return None
            return siman_val[seif - 1]
        if isinstance(siman_val, str):
            return siman_val if seif == 1 else None
        return siman_val

    if isinstance(text_obj, dict):
        preferred_keys: List[str] = []
        if subsection:
            preferred_keys.append(subsection)
        preferred_keys.extend(["", "Main"])

        for key in preferred_keys:
            if key in text_obj:
                value = get_value_at_siman_seif(text_obj[key], siman, seif, subsection=None)
                if value is not None:
                    return value

        for value in text_obj.values():
            found = get_value_at_siman_seif(value, siman, seif, subsection=None)
            if found is not None:
                return found

    return None


def commentary_ref(commentator: str, base_book: str, siman: int, seif: int, idx: int) -> str:
    base = f"{commentator} on {base_book} {siman}:{seif}"
    return base if idx == 1 else f"{base}:{idx}"


def commentary_ref_by_order(source_title: str, siman: int, order: int, idx: int) -> str:
    base = f"{source_title} {siman}:{order}"
    return base if idx == 1 else f"{base}:{idx}"


def extract_commentary_orders(raw_value: Any, commentator: str) -> List[int]:
    pattern = re.compile(
        rf'<i\s+data-commentator="{re.escape(commentator)}"[^>]*data-order="(\d+)"[^>]*></i>'
    )

    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        return [int(match.group(1)) for match in pattern.finditer(raw_value)]

    if isinstance(raw_value, list):
        out: List[int] = []
        for item in raw_value:
            out.extend(extract_commentary_orders(item, commentator))
        return out

    if isinstance(raw_value, dict):
        out: List[int] = []
        for item in raw_value.values():
            out.extend(extract_commentary_orders(item, commentator))
        return out

    return []


def fetch_commentary_refs_from_api(
    session: requests.Session,
    base_ref: str,
    commentator: str,
    timeout_seconds: float = 15.0,
) -> List[str]:
    url = SEFARIA_TEXT_API_TEMPLATE.format(ref=base_ref.replace(" ", "_"))
    try:
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("Failed commentary ref lookup for %s: %s", base_ref, exc)
        return []

    refs: List[str] = []
    for item in data.get("commentary", []) or []:
        collective_title = item.get("collectiveTitle")
        name = collective_title.get("en", "") if isinstance(collective_title, dict) else (collective_title or "")
        if name == commentator:
            ref = (item.get("ref") or "").strip()
            if ref:
                refs.append(ref)
    return refs


def parse_mishnah_berurah_ref(ref: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"Mishnah Berurah\s+(\d+):(\d+)", ref)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def get_mb_text_from_local(
    source_index: Dict[Tuple[str, str], Dict[str, Any]],
    ref: str,
    fallback_siman: int,
) -> str:
    source = source_index.get(("Mishnah Berurah", "Orach Chayim"))
    if not source:
        return ""

    parsed = parse_mishnah_berurah_ref(ref)
    if not parsed:
        return ""

    siman, comment_index = parsed
    if not isinstance(siman, int) or siman <= 0:
        siman = fallback_siman

    raw_value = get_value_at_siman_seif(
        source["data"].get("text"),
        siman=siman,
        seif=comment_index,
        subsection=None,
    )
    segments = flatten_strings(raw_value)
    return segments[0] if segments else ""


def get_commentary_segments_from_local(
    source_index: Dict[Tuple[str, str], Dict[str, Any]],
    commentator: str,
    section: str,
    siman: int,
    order: int,
    subsection: Optional[str] = None,
) -> List[str]:
    source = source_index.get((commentator, section))
    if not source:
        return []

    raw_value = get_value_at_siman_seif(
        source["data"].get("text"),
        siman=siman,
        seif=order,
        subsection=subsection,
    )
    return flatten_strings(raw_value)


def build_rows_for_base(
    base_row: Dict[str, Any],
    source_index: Dict[Tuple[str, str], Dict[str, Any]],
    base_text_index: Dict[str, Dict[str, Any]],
    mb_link_mode: str,
    session: Optional[requests.Session],
) -> List[Dict[str, Any]]:
    metadata = base_row.get("metadata", {}) or {}
    base_ref = base_row.get("sefaria_ref", "")
    base_book = metadata.get("book", "")
    siman = metadata.get("siman")
    seif = metadata.get("seif")
    subsection = metadata.get("subsection")

    if not base_ref or not base_book or not isinstance(siman, int) or not isinstance(seif, int):
        return []

    section = section_from_book(base_book)
    rows: List[Dict[str, Any]] = []
    base_source = base_text_index.get(base_book)
    raw_base_value = None
    if base_source:
        raw_base_value = get_value_at_siman_seif(
            base_source["data"].get("text"),
            siman=siman,
            seif=seif,
            subsection=subsection,
        )

    for commentator in allowed_commentators(base_book):
        if commentator == "Mishnah Berurah" and mb_link_mode == "api_refs" and session is not None:
            refs = fetch_commentary_refs_from_api(session, base_ref, commentator)
            for ref in refs:
                text_he = get_mb_text_from_local(source_index, ref, fallback_siman=siman)
                if not text_he:
                    continue
                rows.append(
                    {
                        "base_ref": base_ref,
                        "commentary_ref": ref,
                        "commentator": commentator,
                        "text_he": text_he,
                        "base_book": base_book,
                        "siman": siman,
                        "seif": seif,
                        "source": "local_merged_json_via_api_ref_map",
                    }
                )
            continue

        if raw_base_value is not None:
            orders = extract_commentary_orders(raw_base_value, commentator)
            if orders:
                source = source_index.get((commentator, section))
                if not source:
                    continue

                for order in orders:
                    segments = get_commentary_segments_from_local(
                        source_index,
                        commentator=commentator,
                        section=section,
                        siman=siman,
                        order=order,
                        subsection=subsection,
                    )
                    for idx, seg in enumerate(segments, start=1):
                        rows.append(
                            {
                                "base_ref": base_ref,
                                "commentary_ref": commentary_ref_by_order(source["title"], siman, order, idx),
                                "commentator": commentator,
                                "text_he": seg,
                                "base_book": base_book,
                                "siman": siman,
                                "seif": seif,
                                "source": "local_merged_json_via_sa_anchor_map",
                            }
                        )
                continue

        source = source_index.get((commentator, section))
        if not source:
            continue

        text_obj = source["data"].get("text")
        raw_value = get_value_at_siman_seif(text_obj, siman=siman, seif=seif, subsection=subsection)
        segments = flatten_strings(raw_value)

        for idx, seg in enumerate(segments, start=1):
            rows.append(
                {
                    "base_ref": base_ref,
                    "commentary_ref": commentary_ref(commentator, base_book, siman, seif, idx),
                    "commentator": commentator,
                    "text_he": seg,
                    "base_book": base_book,
                    "siman": siman,
                    "seif": seif,
                    "source": "local_merged_json",
                }
            )

    return rows


def upsert_rows(supabase: Client, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["base_ref"], row["commentary_ref"])
        deduped[key] = row

    final_rows = list(deduped.values())

    supabase.table("sefaria_commentaries").upsert(
        final_rows,
        on_conflict="base_ref,commentary_ref",
    ).execute()
    return len(final_rows)


def delete_existing_rows(
    supabase: Client,
    book_prefix: str,
) -> None:
    logger.info("Deleting existing commentary rows for %s", book_prefix)
    supabase.table("sefaria_commentaries").delete().like("base_book", f"{book_prefix}%").execute()


def run_backfill(
    local_root: Path,
    page_size: int,
    db_batch_size: int,
    start_offset: int,
    max_base_refs: Optional[int],
    book_prefix: Optional[str],
    mb_link_mode: str,
    delete_existing: bool,
) -> Tuple[int, int]:
    supabase = get_supabase_client()
    source_index = build_source_index(local_root)
    base_text_index = build_base_text_index(local_root)
    session = requests.Session()

    if delete_existing:
        if not book_prefix:
            raise ValueError("--delete-existing requires --book-prefix for safety")
        delete_existing_rows(supabase, book_prefix)

    offset = start_offset
    processed = 0
    upserted = 0
    pending: List[Dict[str, Any]] = []

    while True:
        base_rows = page_base_chunks(
            supabase=supabase,
            page_size=page_size,
            offset=offset,
            book_prefix=book_prefix,
        )
        if not base_rows:
            break

        for base_row in base_rows:
            if max_base_refs is not None and processed >= max_base_refs:
                break

            pending.extend(
                build_rows_for_base(
                    base_row,
                    source_index,
                    base_text_index,
                    mb_link_mode=mb_link_mode,
                    session=session,
                )
            )
            processed += 1

            if len(pending) >= db_batch_size:
                count = upsert_rows(supabase, pending)
                upserted += count
                logger.info("Upserted %d rows (processed refs=%d)", count, processed)
                pending = []

        if max_base_refs is not None and processed >= max_base_refs:
            break

        offset += page_size

    if pending:
        count = upsert_rows(supabase, pending)
        upserted += count
        logger.info("Upserted final %d rows", count)

    return processed, upserted


def detect_local_root(local_root_arg: Optional[str]) -> Path:
    if local_root_arg:
        path = Path(local_root_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"local-root does not exist: {path}")
        return path

    for candidate in DEFAULT_LOCAL_ROOT_CANDIDATES:
        if candidate.exists():
            logger.info("Auto-detected local corpus root: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect local corpus root. Pass --local-root explicitly "
        "(example: c:\\Users\\ympin\\ragbi\\Shulchan-Arukh-Extracted)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill meforshim from local downloaded merged.json files."
    )
    parser.add_argument(
        "--local-root",
        required=False,
        default=None,
        help="Optional root directory containing downloaded Sefaria merged.json files",
    )
    parser.add_argument("--page-size", type=int, default=500, help="Base refs per DB page")
    parser.add_argument("--db-batch-size", type=int, default=400, help="Rows per upsert batch")
    parser.add_argument("--start-offset", type=int, default=0, help="Start offset in sefaria_text_chunks")
    parser.add_argument("--max-base-refs", type=int, default=None, help="Limit refs processed")
    parser.add_argument("--book-prefix", type=str, default=None, help="Optional base book prefix filter")
    parser.add_argument(
        "--mb-link-mode",
        choices=["none", "api_refs"],
        default="api_refs",
        help="How to map Mishnah Berurah to SA seif (api_refs recommended for accuracy)",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing rows for the given --book-prefix before backfilling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_root = detect_local_root(args.local_root)

    start = time.time()
    processed_refs, upserted_rows = run_backfill(
        local_root=local_root,
        page_size=args.page_size,
        db_batch_size=args.db_batch_size,
        start_offset=args.start_offset,
        max_base_refs=args.max_base_refs,
        book_prefix=args.book_prefix,
        mb_link_mode=args.mb_link_mode,
        delete_existing=args.delete_existing,
    )
    elapsed = time.time() - start
    logger.info(
        "Done. processed_refs=%d upserted_rows=%d elapsed=%.2fs",
        processed_refs,
        upserted_rows,
        elapsed,
    )
