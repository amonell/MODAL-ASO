"""Lossless text handling for Excel's per-cell character limit."""

EXCEL_TEXT_UNITS = 32767
PART_TEXT_UNITS = 30000


def excel_text_units(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def excel_text_parts(text: str) -> list[str]:
    """Split at character boundaries, including supplementary Unicode."""
    parts = []
    start = 0
    units = 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xffff else 1
        if units + width > PART_TEXT_UNITS:
            parts.append(text[start:index])
            start = index
            units = 0
        units += width
    parts.append(text[start:])
    return parts


class LosslessExcelValues:
    """Store oversized cells once, with links from every view of a source row."""

    def __init__(self, workbook, columns, reserved_titles=()):
        self.workbook = workbook
        self.columns = list(columns)
        used_titles = {str(name)[:31].casefold() for name in reserved_titles}
        used_titles.update(name.casefold() for name in workbook.sheetnames)
        self.title = 'Long values'
        suffix = 1
        while self.title.casefold() in used_titles:
            suffix += 1
            self.title = f'Long values {suffix}'
        self.values = {}
        self.parts = []

    def append(self, worksheet, source_row_index, columns, values):
        """Append already serialized values; row index refers to the All sheet."""
        cells, links = [], []
        for column_index, (column, value) in enumerate(zip(columns, values), start=1):
            if isinstance(value, str) and excel_text_units(value) > EXCEL_TEXT_UNITS:
                key = (source_row_index, column)
                if key not in self.values:
                    value_id = f'R{source_row_index + 1}C{self.columns.index(column) + 1}'
                    start_row = len(self.parts) + 2
                    parts = excel_text_parts(value)
                    self.values[key] = (value_id, start_row)
                    self.parts.extend([
                        [value_id, source_row_index + 2, column, part_index, len(parts), part]
                        for part_index, part in enumerate(parts, start=1)])
                value_id, start_row = self.values[key]
                value = f'See {self.title}: {value_id}'
                links.append((column_index, start_row))
            cells.append(value)
        worksheet.append(cells)
        for column_index, start_row in links:
            cell = worksheet.cell(worksheet.max_row, column_index)
            cell.hyperlink = f"#'{self.title}'!A{start_row}"
            cell.style = 'Hyperlink'

    def finish(self):
        if not self.parts:
            return
        sheet = self.workbook.create_sheet(title=self.title)
        sheet.append(['Value ID', 'All row', 'Field', 'Part', 'Parts', 'Text'])
        for row in self.parts:
            sheet.append(row)
            # A chunk beginning with '=' is data, never an Excel formula.
            sheet.cell(sheet.max_row, 6).data_type = 's'
        sheet.freeze_panes = 'A2'
