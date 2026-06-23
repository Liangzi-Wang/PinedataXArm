#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def _col_name(idx: int) -> str:
    out = ""
    x = idx
    while x > 0:
        x, rem = divmod(x - 1, 26)
        out = chr(65 + rem) + out
    return out


def _cell_xml(value: str, cell_ref: str) -> str:
    if value is None or value == "":
        return ""
    preserve = ' xml:space="preserve"' if value.startswith(" ") or value.endswith(" ") or "\n" in value else ""
    return f'<c r="{cell_ref}" t="inlineStr"><is><t{preserve}>{escape(value)}</t></is></c>'


def _sheet_xml(rows: list[list[str]]) -> str:
    xml_rows: list[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_col_name(c_idx)}{r_idx}"
            c = _cell_xml(value, ref)
            if c:
                cells.append(c)
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def _write_xlsx(output_path: Path, sheets: list[tuple[str, list[list[str]]]]) -> None:
    content_types_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types_parts.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types_parts.append("</Types>")
    content_types = "".join(content_types_parts)
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_sheets = []
    wb_rels_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (name, _rows) in enumerate(sheets, start=1):
        workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        wb_rels_parts.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    wb_rels_parts.append(
        f'<Relationship Id="rId{len(sheets)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    wb_rels_parts.append("</Relationships>")
    wb_rels = "".join(wb_rels_parts)

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )
    styles = (
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
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/styles.xml", styles)
        for idx, (_name, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def _parse_tab_text(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        cols = [part.strip() for part in raw_line.split("\t")]
        rows.append(cols)
    if not rows:
        return [["(empty)"]]
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def _extract_time_seconds(raw: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _build_time_summary_rows(rows: list[list[str]]) -> list[list[str]]:
    summary = [["metric", "value"]]
    if not rows:
        summary.append(["status", "No rows found"])
        return summary

    header = [str(x).strip().lower() for x in rows[0]]
    try:
        time_col = header.index("time")
    except ValueError:
        summary.append(["status", "No 'time' column found"])
        return summary

    time_values: list[float] = []
    for row in rows[1:]:
        if time_col >= len(row):
            continue
        value = _extract_time_seconds(row[time_col])
        if value is not None:
            time_values.append(value)

    if not time_values:
        summary.append(["status", "No parsable time values found"])
        return summary

    avg_value = sum(time_values) / len(time_values)
    total_value = sum(time_values)
    summary.extend([
        ["parsed_rows", str(len(time_values))],
        ["min_time_s", f"{min(time_values):.2f}"],
        ["max_time_s", f"{max(time_values):.2f}"],
        ["avg_time_s", f"{avg_value:.2f}"],
        ["total_time_s", f"{total_value:.2f}"],
    ])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a tab-separated .txt file to an Excel .xlsx file.")
    parser.add_argument("txt_path", help="Input text file path.")
    parser.add_argument("--output", default="", help="Output .xlsx path (default: same folder, same name).")
    args = parser.parse_args()

    txt_path = Path(args.txt_path).expanduser().resolve()
    if not txt_path.is_file():
        raise SystemExit(f"Input file not found: {txt_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else txt_path.with_suffix(".xlsx")
    rows = _parse_tab_text(txt_path.read_text(encoding="utf-8", errors="replace"))
    summary_rows = _build_time_summary_rows(rows)
    _write_xlsx(output_path, [("data", rows), ("summary", summary_rows)])
    print(f"Excel exported: {output_path}")
    print(f"Rows: {len(rows)}  Columns: {len(rows[0]) if rows else 0}")
    print("Summary sheet: time min/max/avg/total added")


if __name__ == "__main__":
    main()
