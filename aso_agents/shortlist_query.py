"""Read-only, context-preserving queries over a run's immutable ASO evidence."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math

from .shortlist_evidence import digest, fact_value
from .report_attributes import _scalar as scalar_value

VERSION = 'shortlist-query/1'
MAX_RESULT_CHARACTERS = 60000
CONTEXT_KEYS = frozenset({'transcript_id', 'reference_accession', 'species', 'assembly',
    'assembly_name', 'unit', 'units', 'count_unit', 'mismatch_budget', 'mismatch_policy',
    'dosage_nm', 'dose_nm', 'transfection_method', 'delivery', 'model', 'model_id',
    'perturbation', 'perturbation_type', 'site_kind', 'junction_kind', 'target_role',
    'context_basis', 'kind', 'expression_tissue', 'Off_target_expression_tissue'})
UNAVAILABLE = frozenset({'failed', 'error', 'unavailable', 'not_run', 'unsupported',
    'incomplete', 'incomplete_or_unavailable', 'incomplete_or_censored', 'skipped'})


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


@lru_cache(maxsize=32768)
def normalized_path(pointer, literal_last=False):
    parts = pointer.split('/')
    return '/'.join('[]' if token.isdecimal() and not (literal_last and i == len(parts)-1)
                    else token for i, token in enumerate(parts))


@lru_cache(maxsize=32768)
def attribute_id(path):
    return 'a_' + digest(path)[:16]


def atoms(facts):
    for pointer, value in facts.items():
        if isinstance(value, (list, tuple)):
            converted = [scalar_value(item, tuple(pointer.split('/'))) for item in value]
            numeric_array = all(item is None or number(item) for item in converted)
            for index, item in enumerate(converted):
                if item is None or isinstance(item, (str, bool)) or number(item):
                    yield pointer + '/' + str(index), item, numeric_array
        elif value is None or isinstance(value, (str, bool)) or number(value):
            # A terminal numeric key in flattened object facts is a named bin.
            # Scalar arrays are handled above with their known container type.
            yield pointer, scalar_value(value, tuple(pointer.split('/'))), pointer.rsplit('/', 1)[-1].isdecimal()


def kind(value):
    return 'number' if number(value) else 'boolean' if isinstance(value, bool) else 'category'


def bounded(value):
    if len(json.dumps(value, ensure_ascii=False, separators=(',', ':'))) > MAX_RESULT_CHARACTERS:
        raise ValueError('Result exceeds the response limit; request fewer fields or a smaller page.')
    return value


def _condition(value, operator, expected):
    if operator == 'exists':
        if not isinstance(expected, bool):
            raise ValueError('An existence comparison requires true or false.')
        return (value is not None) == expected
    if value is None:
        return None
    if operator in {'lt', 'lte', 'gt', 'gte', 'between'}:
        if not number(value):
            raise ValueError('Numeric comparisons require finite numeric evidence.')
        limits = expected if operator == 'between' else [expected]
        if not isinstance(limits, list) or not all(number(item) for item in limits):
            raise ValueError('Numeric comparisons require finite numeric thresholds.')
        if operator == 'between':
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError('A range requires ordered lower and upper thresholds.')
            return limits[0] <= value <= limits[1]
        return {'lt': value < expected, 'lte': value <= expected,
                'gt': value > expected, 'gte': value >= expected}[operator]
    if operator in {'eq', 'ne'}:
        equal = type(value) is type(expected) and value == expected or number(value) and number(expected) and value == expected
        return equal if operator == 'eq' else not equal
    if operator in {'in', 'not_in'}:
        if not isinstance(expected, list):
            raise ValueError('Membership requires a list of literal values.')
        found = any(_condition(value, 'eq', item) for item in expected)
        return found if operator == 'in' else not found
    raise ValueError('Unknown comparison operator.')


class CandidateIndex:
    """Index field locations once; materialize requested columns lazily.

    Array records keep their source indices and sibling conditions. Numeric
    scalar arrays retain element indices (for example mismatch-count bins).
    No ordering direction or biological threshold is inferred from a label.
    """
    def __init__(self, catalog, reserve_ids=()):
        self.catalog = {key: entry for key, entry in catalog.items() if entry['eligible']}
        self.ids = tuple(self.catalog)
        self.fields, self._columns, self._metadata = {}, {}, {}
        self.sets = {'all': frozenset(self.ids)}
        self.ordered_sets = {'all': self.ids}
        self.reserve_ids = set(reserve_ids) & set(self.ids)
        self.compared_ids, self.read_ids = set(), set()
        self.operations = []
        self._groups = {}
        for key, entry in self.catalog.items():
            seen = set()
            for pointer, value, literal in atoms(entry['facts']):
                root = '/'.join(pointer.split('/')[:3])
                if entry['facts'].get(root + '/candidate_assessment/selection_eligible') is not True:
                    continue
                path = normalized_path(pointer, literal)
                field_id = attribute_id(path)
                field = self.fields.setdefault(field_id, {'id': field_id, 'path': path,
                    'sources': set(), 'kinds': set(), 'available_molecule_count': 0})
                field['sources'].add(pointer)
                if value is not None:
                    field['kinds'].add(kind(value))
                    seen.add(field_id)
            for field_id in seen:
                self.fields[field_id]['available_molecule_count'] += 1
            groups = set()
            for overview in entry.get('overview', []):
                geometry = overview.get('binding_and_objective') or {}
                for name in ('program_id', 'job_id', 'site_label', 'target_region_label', 'route_label'):
                    if geometry.get(name):
                        groups.add(name + ': ' + str(geometry[name]))
            self._groups[key] = groups

    def field(self, field_id):
        if field_id not in self.fields:
            raise ValueError('Unknown attribute ID; discover IDs with survey.')
        return self.fields[field_id]

    def _meta(self, key):
        if key not in self._metadata:
            meta, statuses = defaultdict(dict), {}
            for pointer, value in self.catalog[key]['facts'].items():
                parent, _, name = pointer.rpartition('/')
                if name in CONTEXT_KEYS and (isinstance(value, (str, bool)) or number(value)):
                    meta[parent][normalized_path(pointer)] = value
                if '/conditions/' in pointer and (isinstance(value, (str, bool)) or number(value)):
                    owner = pointer.split('/conditions/', 1)[0]
                    meta[owner][normalized_path(pointer)] = value
                if name == 'status' and isinstance(value, str):
                    statuses[parent] = value
            self._metadata[key] = meta, statuses
        return self._metadata[key]

    def column(self, field_id):
        if field_id in self._columns:
            return self._columns[field_id]
        field = self.field(field_id)
        values, scopes = {}, {}
        for key, entry in self.catalog.items():
            observations = []
            meta, statuses = self._meta(key)
            for pointer in sorted(field['sources']):
                try:
                    value = fact_value(entry, pointer)
                except KeyError:
                    continue
                parts = pointer.split('/')
                value = scalar_value(value, tuple(part for part in parts if not part.isdecimal()))
                root = '/'.join(parts[:3])
                if entry['facts'].get(root + '/candidate_assessment/selection_eligible') is not True:
                    continue
                scope, bindings, invalid = {}, {}, False
                for end in range(3, len(parts)):
                    parent = '/'.join(parts[:end])
                    scope.update(meta.get(parent, {}))
                    invalid |= statuses.get(parent) in UNAVAILABLE
                    if parts[end-1].isdecimal():
                        bindings[normalized_path('/'.join(parts[:end-1]))] = int(parts[end-1])
                name = parts[-1].casefold()
                if number(value) and 'off_target' in name and ('count' in name or 'risk' in name):
                    canonical = root + '/candidate_assessment/off_target'
                    scope.update(meta.get(canonical, {}))
                    invalid |= statuses.get(canonical) not in {'complete', 'complete_within_declared_scope'}
                if number(value) and name.startswith('oligoai'):
                    canonical = root + '/oligoai_evaluation'
                    scope.update(meta.get(canonical, {}))
                    invalid |= statuses.get(canonical) in UNAVAILABLE
                scope_id = 's_' + digest(scope)[:16]
                scopes[scope_id] = scope
                # A failed measurement with a residual numeric value stays
                # unavailable. Its original value remains in read_evidence.
                observations.append({'value': None if invalid and number(value) else value,
                    'path': pointer, 'scope_id': scope_id, 'bindings': bindings,
                    'context': parts[2], 'unavailable': invalid})
            if observations:
                values[key] = observations
        result = {'values': values, 'scopes': scopes}
        self._columns[field_id] = result
        return result

    def survey(self, field_ids=None, offset=0, limit=24, scope_offset=0, terms=None):
        if offset < 0 or scope_offset < 0 or not 1 <= limit <= 40:
            raise ValueError('Use a nonnegative offset and a page size from 1 to 40.')
        if not field_ids:
            chosen = self.discover(terms) if terms else self.fields
            fields = sorted((self.fields[key] for key in chosen), key=lambda item: item['path'])
            page = fields[offset:offset+limit]
            return {'molecule_count': len(self.ids), 'attribute_count': len(fields),
                'attributes': [{name: field[name] for name in ('id', 'path', 'available_molecule_count')}
                    | {'kinds': sorted(field['kinds'])} for field in page],
                'next_offset': offset+len(page) if offset+len(page) < len(fields) else None,
                'constraint_reserve_count': len(self.reserve_ids)}
        if len(field_ids) > 12:
            raise ValueError('Survey at most 12 attributes at once.')
        result = []
        for field_id in field_ids:
            column, summaries = self.column(field_id), []
            scope_ids = sorted(column['scopes'])
            for scope_id in scope_ids[scope_offset:scope_offset+limit]:
                numbers, categories, observed = [], Counter(), set()
                for key, items in column['values'].items():
                    for item in items:
                        if item['scope_id'] != scope_id or item['value'] is None:
                            continue
                        observed.add(key)
                        if number(item['value']):
                            numbers.append(item['value'])
                        else:
                            categories[str(item['value'])] += 1
                summary = {'scope_id': scope_id, 'conditions': column['scopes'][scope_id],
                    'available_molecule_count': len(observed), 'missing_molecule_count': len(self.ids)-len(observed),
                    'observation_count': len(numbers)+sum(categories.values()),
                    'distribution_basis': 'Individual recorded observations; a molecule can contribute several observations.'}
                if numbers:
                    numbers.sort()
                    summary['quantiles'] = {str(q): numbers[round(q*(len(numbers)-1))] for q in (0, .1, .25, .5, .75, .9, 1)}
                if categories:
                    summary['categories'] = categories.most_common(12)
                summaries.append(summary)
            result.append({'id': field_id, 'path': self.field(field_id)['path'], 'scopes': summaries,
                'scope_count': len(scope_ids), 'next_scope_offset': scope_offset+len(summaries)
                    if scope_offset+len(summaries) < len(scope_ids) else None})
        return bounded({'molecule_count': len(self.ids), 'attributes': result})

    def discover(self, terms):
        """Literal word discovery; returned IDs still control every operation."""
        words = [word.casefold() for word in terms]
        return [key for key, field in self.fields.items()
                if any(word in field['path'].casefold() for word in words)]

    def _scopes(self, predicate):
        column = self.column(predicate['field_id'])
        chosen = predicate['scope_ids']
        if not chosen:
            if len(column['scopes']) > 1:
                raise ValueError('This attribute has multiple conditions; choose explicit scope_ids from survey.')
            chosen = list(column['scopes'])
        if any(key not in column['scopes'] for key in chosen):
            raise ValueError('Unknown scope ID for this attribute.')
        # Transcript/species aggregations are explicit. Dose, delivery, units
        # and model settings remain identical within a single comparison.
        comparability = []
        for key in chosen:
            comparability.append({path: value for path, value in column['scopes'][key].items()
                if '/conditions/' in path or path.rsplit('/', 1)[-1] in {
                    'unit', 'units', 'count_unit', 'mismatch_budget', 'mismatch_policy',
                    'dosage_nm', 'dose_nm', 'transfection_method', 'delivery', 'model', 'model_id',
                    'assembly', 'assembly_name'}})
        if len({digest(value) for value in comparability}) > 1:
            raise ValueError('Compare different units, doses, delivery methods or search scopes in separate queries.')
        return set(chosen)

    def _predicate(self, key, context, predicate, scopes):
        observations = [item for item in self.column(predicate['field_id'])['values'].get(key, [])
            if item['scope_id'] in scopes and item['context'] == context]
        values = [item['value'] for item in observations]
        if scopes-set(item['scope_id'] for item in observations):
            values.append(None)
        if not values and predicate['operator'] == 'exists':
            values = [None]
        aggregation = predicate['aggregation']
        operator, expected = predicate['operator'], predicate['value']
        if aggregation in {'min', 'max'}:
            if not values or any(value is None for value in values):
                status = None
            elif not all(number(value) for value in values):
                raise ValueError('Minimum/maximum require numeric observations.')
            else:
                status = _condition((min if aggregation == 'min' else max)(values), operator, expected)
            matches = [{}] if status is True else []
        else:
            results = [_condition(value, operator, expected) for value in values]
            if aggregation == 'any':
                status = True if True in results else None if not results or None in results else False
                matches = [item['bindings'] for item, value in zip(observations, results) if value is True]
                if operator == 'exists' and True in results[len(observations):]:
                    matches = [{}]
            elif aggregation == 'all':
                status = False if False in results else None if not results or None in results else True
                matches = [{}] if status is True else []
            else:
                raise ValueError('Choose any, all, min or max aggregation.')
        missing = predicate['missing']
        if missing not in {'include', 'exclude', 'only'}:
            raise ValueError('Choose explicit missing-value handling.')
        if missing == 'only':
            return ([{}] if status is None else []), status is None
        if status is None and missing == 'include':
            matches = [{}]
        return matches, status is None

    def coverage(self, ids):
        groups = Counter(group for key in ids for group in self._groups[key])
        return {'molecule_count': len(ids), 'constraint_reserve_retained': len(set(ids) & self.reserve_ids),
                'constraint_reserve_omitted': len(self.reserve_ids-set(ids)), 'groups': dict(sorted(groups.items()))}

    def filter(self, input_sets, combine, predicates, join, context_policy, rationale, offset=0, limit=64, order_by=None):
        if not input_sets or any(key not in self.sets for key in input_sets):
            raise ValueError('Choose existing candidate set IDs, starting with all.')
        if not 1 <= limit <= 128 or offset < 0 or len(predicates) > 12 or not rationale.strip():
            raise ValueError('Supply a rationale, at most 12 predicates, and a page size from 1 to 128.')
        groups = [self.sets[key] for key in input_sets]
        if combine == 'union':
            starting = set().union(*groups)
        elif combine == 'intersection':
            starting = set(groups[0]).intersection(*groups[1:])
        elif combine == 'difference':
            starting = set(groups[0]).difference(*groups[1:])
        else:
            raise ValueError('Choose union, intersection or difference.')
        if join not in {'all', 'any'} or context_policy not in {'any', 'all'}:
            raise ValueError('Choose all or any for predicate and context combination.')
        scopes = [self._scopes(predicate) for predicate in predicates]
        matched, unknown = [], []
        for key in self.ids:
            if key not in starting:
                continue
            contexts = [str(item['context']) for item in self.catalog[key]['overview']
                if self.catalog[key]['facts'].get('/contexts/'+str(item['context'])+'/candidate_assessment/selection_eligible') is True]
            results, unresolved = [], False
            for context in contexts:
                candidates = [{}]
                any_match = False
                for predicate, scope_ids in zip(predicates, scopes):
                    matches, missing = self._predicate(key, context, predicate, scope_ids)
                    unresolved |= missing
                    any_match |= bool(matches)
                    if join == 'all':
                        compatible = []
                        for existing in candidates:
                            for binding in matches:
                                if all(name not in existing or existing[name] == value for name, value in binding.items()):
                                    compatible.append({**existing, **binding})
                        # De-duplicate binding assignments rather than building
                        # a Cartesian product of unrelated record values.
                        candidates = list({tuple(sorted(item.items())): item for item in compatible}.values())
                        if len(candidates) > 4096:
                            raise ValueError('Narrow the record scopes before combining these predicates.')
                results.append(bool(candidates) if join == 'all' else any_match)
            accepted = (all(results) if context_policy == 'all' else any(results)) if predicates else True
            if accepted:
                matched.append(key)
            if unresolved:
                unknown.append(key)
        order_by = order_by or []
        if len(order_by) > 4:
            raise ValueError('Use at most four explicit sorting criteria.')
        for criterion in reversed(order_by):
            chosen = self._scopes(criterion)
            if criterion['aggregation'] not in {'min', 'max'} or criterion['direction'] not in {'ascending', 'descending'}:
                raise ValueError('Numeric sorting requires min/max and ascending/descending.')
            column = self.column(criterion['field_id'])
            def sort_value(key):
                items = [item for item in column['values'].get(key, []) if item['scope_id'] in chosen]
                values = [item['value'] for item in items]
                if not values or any(not number(value) for value in values) or chosen-set(item['scope_id'] for item in items):
                    return (1, 0)
                value = (min if criterion['aggregation'] == 'min' else max)(values)
                return (0, -value if criterion['direction'] == 'descending' else value)
            matched.sort(key=sort_value)
        query = {'input_sets': input_sets, 'combine': combine, 'predicates': predicates,
                 'join': join, 'context_policy': context_policy, 'order_by': order_by}
        set_id = 'q_' + digest(query)[:16]
        operation = {'set_id': set_id, 'query': query, 'rationale': rationale,
            'input_count': len(starting), 'matched_count': len(matched), 'unknown_count': len(unknown),
            'excluded_count': len(starting)-len(matched), 'coverage_before': self.coverage(starting),
            'coverage_after': self.coverage(matched)}
        result = bounded({**operation, 'candidate_ids': matched[offset:offset+limit],
            'next_offset': offset+limit if offset+limit < len(matched) else None})
        self.sets[set_id] = frozenset(matched)
        self.ordered_sets[set_id] = tuple(matched)
        if not any(item['set_id'] == set_id for item in self.operations):
            self.operations.append(operation)
        return result

    def compare(self, candidate_ids, field_ids, offset=0, limit=16):
        if any(key not in self.catalog for key in candidate_ids) or len(field_ids) > 12:
            raise ValueError('Use known candidate IDs and at most 12 attributes.')
        if not 1 <= limit <= 32 or offset < 0:
            raise ValueError('Use a nonnegative offset and a page size from 1 to 32.')
        rows = []
        for key in candidate_ids[offset:offset+limit]:
            row = {'id': key, 'identity': self.catalog[key]['identity'],
                'overview': self.catalog[key]['overview'], 'attributes': {
                    field_id: [{**item, 'conditions': self.column(field_id)['scopes'][item['scope_id']]}
                        for item in self.column(field_id)['values'].get(key, [])] for field_id in field_ids}}
            if len(json.dumps(rows+[row], separators=(',', ':'))) > MAX_RESULT_CHARACTERS-1000:
                if not rows:
                    raise ValueError('This record requires read_evidence pages or fewer attributes.')
                break
            rows.append(row)
        self.compared_ids.update(row['id'] for row in rows)
        end = offset+len(rows)
        return bounded({'candidates': rows, 'next_offset': end if end < len(candidate_ids) else None})

    def read_evidence(self, candidate_id, page=0):
        from .shortlist_paging import evidence_fragments, evidence_pages
        if candidate_id not in self.catalog:
            raise ValueError('Unknown candidate ID.')
        pages = evidence_pages(evidence_fragments(self.catalog[candidate_id]['facts']),
                               limit=MAX_RESULT_CHARACTERS-1000)
        if page < 0 or page >= len(pages):
            raise ValueError('Choose a valid evidence page.')
        self.read_ids.add(candidate_id)
        return {'candidate_id': candidate_id, 'page': page, 'page_count': len(pages),
                'source_sha256': digest(self.catalog[candidate_id]['facts']), 'records': pages[page],
                'next_page': page+1 if page+1 < len(pages) else None}

    def outsiders(self, selected, limit=32):
        """A deterministic, auditable challenge set from the excluded library."""
        excluded = set(self.ids)-set(selected)
        priorities = []
        # Original native order and every recorded group provide independent
        # starting points, including groups lost by proposed filters.
        priorities.extend(key for key in self.ids if key in self.reserve_ids & excluded)
        groups_seen = set()
        for key in self.ids:
            if key in excluded and self._groups[key]-groups_seen:
                priorities.append(key)
                groups_seen.update(self._groups[key])
        for operation in self.operations:
            for predicate in operation['query']['predicates']:
                column = self.column(predicate['field_id'])
                scopes = self._scopes(predicate)
                values, missing = [], []
                for key in excluded:
                    items = [item['value'] for item in column['values'].get(key, []) if item['scope_id'] in scopes]
                    valid = [value for value in items if number(value)]
                    if valid:
                        values.append((min(valid), max(valid), key))
                    else:
                        missing.append(key)
                values.sort()
                priorities.extend(item[2] for item in values[:2]+values[-2:])
                expected = predicate['value']
                if number(expected):
                    priorities.extend(item[2] for item in sorted(values,
                        key=lambda item: (min(abs(item[0]-expected), abs(item[1]-expected)), item[2]))[:2])
                priorities.extend(sorted(missing)[:2])
        # Hash order supplies reproducible representatives away from the
        # original ordering and filter thresholds.
        priorities.extend(sorted(excluded, key=lambda key: digest(key)))
        return list(dict.fromkeys(priorities))[:limit]
