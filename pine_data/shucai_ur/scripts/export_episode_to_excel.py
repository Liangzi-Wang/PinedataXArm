#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np


def _col_name(idx: int) -> str:
    name = ""
    x = idx
    while x > 0:
        x, rem = divmod(x - 1, 26)
        name = chr(65 + rem) + name
    return name


def _safe_sheet_name(name: str, used: set[str]) -> str:
    banned = set(r'[]:*?/\\')
    cleaned = "".join("_" if ch in banned else ch for ch in name).strip() or "Sheet"
    cleaned = cleaned[:31]
    base = cleaned
    i = 1
    while cleaned in used:
        suffix = f"_{i}"
        cleaned = f"{base[:31-len(suffix)]}{suffix}"
        i += 1
    used.add(cleaned)
    return cleaned


def _format_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _flatten_json(prefix: str, value: Any, out_rows: list[list[Any]]) -> None:
    value = _format_scalar(value)
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_json(next_prefix, v, out_rows)
        return
    if isinstance(value, list):
        if len(value) == 0:
            out_rows.append([prefix, "[]"])
            return
        if all(not isinstance(item, (dict, list)) for item in value):
            out_rows.append([prefix, json.dumps([_format_scalar(x) for x in value], ensure_ascii=False)])
            return
        for i, item in enumerate(value):
            _flatten_json(f"{prefix}[{i}]", item, out_rows)
        return
    out_rows.append([prefix, value])


def _to_rows_2d(arr: np.ndarray) -> list[list[Any]]:
    arr2 = np.asarray(arr)
    if arr2.ndim == 1:
        rows = [["index", "value"]]
        for i, v in enumerate(arr2.tolist()):
            rows.append([i, _format_scalar(v)])
        return rows
    if arr2.ndim == 2:
        rows = [["index"] + [f"c{j}" for j in range(arr2.shape[1])]]
        for i in range(arr2.shape[0]):
            rows.append([i] + [_format_scalar(x) for x in arr2[i].tolist()])
        return rows
    return []


def _cell_xml(value: Any, cell_ref: str) -> str:
    value = _format_scalar(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            txt = "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
            return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(txt)}</t></is></c>'
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = str(value)
    # Preserve newlines/spaces where needed.
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") or "\n" in text else ""
    return f'<c r="{cell_ref}" t="inlineStr"><is><t{preserve}>{escape(text)}</t></is></c>'


def _sheet_xml(rows: list[list[Any]]) -> str:
    xml_rows: list[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for c_idx, value in enumerate(row, start=1):
            cell_ref = f"{_col_name(c_idx)}{r_idx}"
            cell = _cell_xml(value, cell_ref)
            if cell:
                cells.append(cell)
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def _write_xlsx(output_path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    wb_sheets = []
    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i, (name, _rows) in enumerate(sheets, start=1):
        wb_sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        wb_rels.append(
            f'<Relationship Id="rId{i}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    wb_rels.append(
        f'<Relationship Id="rId{len(sheets)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    wb_rels.append("</Relationships>")

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(wb_sheets)}</sheets>"
        "</workbook>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        "</styleSheet>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels))
        zf.writestr("xl/styles.xml", styles_xml)
        for i, (_name, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))


def build_workbook_data(episode_dir: Path) -> list[tuple[str, list[list[Any]]]]:
    sheets: list[tuple[str, list[list[Any]]]] = []
    used_sheet_names: set[str] = set()

    metadata_path = episode_dir / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata = {"metadata_error": str(exc)}

    metadata_rows: list[list[Any]] = [["key", "value"]]
    _flatten_json("", metadata, metadata_rows)
    sheets.append((_safe_sheet_name("metadata", used_sheet_names), metadata_rows))

    files_rows: list[list[Any]] = [["file", "size_bytes", "dtype", "shape", "note"]]
    npy_files = sorted(episode_dir.glob("*.npy"))
    for npy_path in npy_files:
        note = ""
        dtype = ""
        shape: Any = ""
        try:
            arr = np.load(npy_path, mmap_mode="r")
            dtype = str(arr.dtype)
            shape = list(arr.shape)
            if arr.ndim > 2:
                note = "Skipped: array has >2 dimensions (not tabular)."
            else:
                rows = _to_rows_2d(np.asarray(arr))
                if rows:
                    sheets.append((_safe_sheet_name(npy_path.stem, used_sheet_names), rows))
        except Exception as exc:
            note = f"Read error: {exc}"

        files_rows.append([npy_path.name, npy_path.stat().st_size, dtype, json.dumps(shape), note])

    for video in sorted(list(episode_dir.glob("*.mkv")) + list(episode_dir.glob("*.mp4"))):
        files_rows.append([video.name, video.stat().st_size, "", "", "Video stream (kept as file, not expanded)."])

    sheets.insert(1, (_safe_sheet_name("files_summary", used_sheet_names), files_rows))
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one recorded episode folder into an Excel .xlsx workbook.")
    parser.add_argument("episode_dir", help="Episode directory path (contains metadata.json / npy / mkv).")
    parser.add_argument(
        "--output",
        default="",
        help="Output .xlsx path (default: <episode_dir>/<episode_name>_export.xlsx).",
    )
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    if not episode_dir.is_dir():
        raise SystemExit(f"Episode directory not found: {episode_dir}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = episode_dir / f"{episode_dir.name}_export.xlsx"

    sheets = build_workbook_data(episode_dir)
    _write_xlsx(output_path, sheets)
    print(f"Excel exported: {output_path}")
    print(f"Sheets: {[name for name, _ in sheets]}")


if __name__ == "__main__":
    main()
