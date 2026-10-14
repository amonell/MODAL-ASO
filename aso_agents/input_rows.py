"""Identity and provenance for supplied observations, independent of sequence rank."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import unicodedata


def _file_expansion_identity(row):
    """Compare formatting-equivalent chemistry without changing source text."""
    chemistry = unicodedata.normalize('NFKC', str(row.get('chemistry') or ''))
    chemistry = chemistry.translate(str.maketrans('', '', "'’′\u2032`"))
    chemistry = ' '.join(chemistry.casefold().split())
    return (row.get('name'), row.get('sequence'), chemistry)


def reconcile_file_expansions(rows, file_count, request_text):
    """Retain physical file rows; reconcile a parser's redundant file expansion.

    A molecule literally present in the request, an explicitly inline observation,
    or a row with additional/conflicting metadata stays separate. With no original
    request available, do not infer provenance. Copies are retained as provenance
    records instead of being counted as additional submitted observations.
    """
    files, inline = rows[:file_count], rows[file_count:]
    index = defaultdict(list)
    # Platform labels and narrative notes can be added by the planner when it
    # expands an attached table; they do not create a second physical record.
    for row in files:
        index[_file_expansion_identity(row)].append(row)
    result = list(files)
    for row in inline:
        matches = index.get(_file_expansion_identity(row), [])
        sequence = str(row.get('sequence') or '')
        literal = bool(sequence and sequence.upper() in str(request_text or '').upper())
        extra = row.get('input_metadata') or {}
        compatible = [m for m in matches if all((m.get('input_metadata') or {}).get(k) == v
                                                for k, v in extra.items())]
        if not request_text or literal or row.get('input_origin') == 'inline_user' or not compatible:
            result.append(row)
            continue
        alias = {'kind': 'parsed_file_row_alias', 'basis': 'same name, sequence, chemistry and metadata; absent from original request literals',
                 'parsed_row': deepcopy(row), 'matching_source_rows': [
                     {'source_file': m.get('source_file'), 'source_row': m.get('source_row')}
                     for m in compatible]}
        # Do not deduplicate real file rows, even if all their fields are equal.
        compatible[0].setdefault('input_reconciliation', []).append(alias)
    return result


def assign_input_row_ids(rows):
    """Stable identity for every observation, including repeated names/sequences."""
    occurrences = defaultdict(int)
    for index, row in enumerate(rows):
        origin = ({'source_file': row['source_file'], 'source_row': row.get('source_row')}
                  if row.get('source_file') else {'inline_position': index})
        key = json.dumps(origin, sort_keys=True, ensure_ascii=False)
        occurrences[key] += 1
        token = hashlib.sha256((key + ':' + str(occurrences[key])).encode()).hexdigest()[:20]
        row['input_row_id'] = 'input_' + token
    return rows


def assessment_row_lookup(rows):
    """Index observations once; names are a legacy fallback only if unique."""
    by_id, by_name = defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get('input_row_id'):
            by_id[row['input_row_id']].append(row)
        by_name[row.get('name')].append(row)

    def resolve(assessment):
        if not isinstance(assessment, dict):
            return None
        row_id = assessment.get('input_row_id')
        matches = by_id.get(row_id, []) if row_id else by_name.get(assessment.get('name'), [])
        if len(matches) == 1 and assessment.get('name') == matches[0].get('name'):
            return matches[0]
        return None
    return resolve


def assessment_coverage_errors(assessments, rows):
    """Check row attribution and delivery, without grading any scientific verdict."""
    errors, found = [], defaultdict(int)
    resolve = assessment_row_lookup(rows)
    for index, item in enumerate(assessments):
        row = resolve(item)
        if row is None:
            errors.append(f'evaluation_assessments[{index}]: unknown or ambiguous row; copy its exact input_row_id and name.')
        else:
            found[id(row)] += 1
    for row in rows:
        if found[id(row)] != 1:
            errors.append(f"Provide exactly one assessment for input_row_id={row.get('input_row_id')!r}, name={row.get('name')!r}; received {found[id(row)]}.")
    return errors
