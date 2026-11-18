"""Source-backed phrases for exact quantities in a recommendation."""
from .shortlist_evidence import _leaves, fact_value


def binding_sources(row, geometry, context):
    """Identify the original inputs used by the coordinate exporter."""
    from .report_explorer_data import collection, source_row
    source = source_row(row)
    prefix = f'/contexts/{context}/'
    if len(geometry.get('binding_intervals_1based') or []) != 1 or geometry.get('strand') not in ('+', '-'):
        return []
    def source_path(name):
        return prefix + ('' if name in row else 'source_annotations/') + name
    strand = next(name for name in ('strand', 'Strand', 'Genomic_strand') if source.get(name))
    paths = [source_path(strand)]
    if isinstance(collection(row.get('binding_intervals_bed0_half_open')), list):
        return paths + [prefix + 'binding_intervals_bed0_half_open']
    convention = next((name for name in ('coordinate_system', 'genomic_coordinate_system') if source.get(name)), None)
    if convention:
        paths.append(source_path(convention))
    segments = next((name for name in ('Genomic_segments', 'genomic_segments') if source.get(name)), None)
    if segments and isinstance(collection(source[segments]), list) and convention and source[convention] in (
            '1based_inclusive', 'genomic_1based_inclusive'):
        return paths + [source_path(segments) + suffix for suffix, _ in _leaves(source[segments])]
    for primary, alternate in [('Chrom', 'chrom'), ('Start', 'start'), ('End', 'end')]:
        name = primary if (source.get(primary) if primary == 'Chrom' else primary in source) else alternate
        paths.append(source_path(name))
    return paths


def verified_phrases(catalog, ids, geometry):
    """Only offer phrases whose underlying values resolve on the original rows."""
    phrases = {}
    def add(owner, text, refs):
        if not refs:
            return
        try:
            for key, path in refs:
                fact_value(catalog[key], path)
        except KeyError:
            return
        token = '[[F' + str(len(phrases) + 1) + ']]'
        phrases[token] = {'candidate_id': owner, 'text': text,
                          'sources': [{'candidate_id': key, 'path': path} for key, path in dict.fromkeys(refs)]}
    for key in ids:
        entry = catalog[key]
        sources = entry.get('binding_sources', [])
        # The complete geometry still goes to the reviewer. Bound optional
        # writing aids for molecules with many overlapping target definitions.
        for observation in entry.get('coordinate_observations', [])[:6]:
            ctx = observation['context']
            if ctx >= len(sources) or not sources[ctx]:
                continue
            member = f"/contexts/{ctx}/source_annotations/binding_site_membership/{observation['source_membership_index']}/"
            label = next((member + name for name in ('label', 'constraint_label')
                          if entry['facts'].get(member + name) == observation['target_label']
                          and isinstance(observation['target_label'], str) and observation['target_label'].strip()
                          and len(observation['target_label']) <= 100 and '[[' not in observation['target_label']
                          and ']]' not in observation['target_label']), None)
            target = observation['target_label'] if label else 'the target'
            refs = [(key, path) for path in [*sources[ctx], member + 'genomic_segments_bed0_half_open']]
            if label:
                refs.append((key, label))
            first, last = observation['target_bases_covered_1based_in_RNA_order']
            covered = f'base {first}' if first == last else f'bases {first}–{last}'
            add(key, f'covers {covered} of {target} in RNA order', refs)
            upstream = observation['bases_beyond_target_upstream_in_RNA']
            downstream = observation['bases_beyond_target_downstream_in_RNA']
            extensions = [f'{size} nt {side}' for size, side in ((upstream, 'upstream'), (downstream, 'downstream')) if size]
            if extensions:
                add(key, 'extends ' + ' and '.join(extensions) + ' beyond ' + target, refs)
        placement = geometry.get(key, {})
        anchor = placement.get('relative_to_candidate_id')
        if anchor in catalog and anchor != key and sources and sources[0]:
            anchor_sources = catalog[anchor].get('binding_sources', [])
            if anchor_sources and anchor_sources[0]:
                refs = [(key, path) for path in sources[0]] + [(anchor, path) for path in anchor_sources[0]]
                shift = placement['start_shift_genomic_bp']
                if shift:
                    add(key, f"starts {abs(shift):,} bp {placement['start_direction_from_anchor']} of ASO {placement['relative_to_panel_order']} on the genome", refs)
        # Multiple contexts can contain different searches for one molecule.
        # Leave those original scoped records in the evidence packet.
        if len(entry.get('overview', [])) != 1:
            continue
        screen = entry['overview'][0].get('off_target') or {}
        budget = screen.get('mismatch_budget')
        units = {'loci': ('locus', 'loci'), 'genes': ('gene', 'genes'), 'sites': ('site', 'sites'),
                 'transcripts': ('transcript', 'transcripts')}.get(screen.get('unit'))
        if screen.get('status') != 'complete_within_declared_scope' or type(budget) is not int or not units:
            continue
        prefix = '/contexts/0/candidate_assessment/off_target/'
        for mismatch, count in enumerate(screen.get('counts_by_mismatch') or []):
            if (mismatch > budget or type(count) not in (int, float) or count < 0 or int(count) != count):
                continue
            how = ('with an exact sequence match' if mismatch == 0 else
                   'with exactly one mismatch' if mismatch == 1 else f'with exactly {mismatch} mismatches')
            add(key, f"the screen found {int(count)} {units[count != 1]} {how}", [(key, prefix + suffix) for suffix in
                (f'counts_by_mismatch/{mismatch}', 'mismatch_budget', 'unit', 'status')])
    return phrases


def render_phrases(item, phrases):
    """Expand only this molecule's handles and retain exact comparison citations."""
    errors = []
    owner = item['candidate_id']
    for field in ('why_chosen', 'tradeoff', 'panel_role'):
        value = item[field]
        parts = value.split('[[')
        if ']]' in parts[0]:
            errors.append(f'{field} contains a malformed verified fact.')
        output = [parts[0]]
        for part in parts[1:]:
            token, end, tail = part.partition(']]')
            fact = phrases.get('[[' + token + ']]')
            if not end or ']]' in tail or not fact or fact['candidate_id'] != owner:
                errors.append(f'{field} requires an exact verified fact handle belonging to this ASO.')
                continue
            phrase = fact['text']
            if not ''.join(output).strip():
                phrase = phrase[0].upper() + phrase[1:]
            output.extend((phrase, tail))
            for ref in fact['sources']:
                if ref['candidate_id'] == owner:
                    item['evidence_paths'].append(ref['path'])
                elif ref not in item['comparison_evidence']:
                    item['comparison_evidence'].append(dict(ref))
        item[field] = ''.join(output)
        if len(item[field]) > 450:
            errors.append(f'Keep the rendered {field} within 450 characters.')
    return errors


def sourced_audit_edit(draft, audit, catalog, expand_reference):
    """Apply only unanimous, source-resolved wording edits to explicitly rejected fields."""
    from copy import deepcopy
    if not audit['issues'] or any(finding.get('selection_impact') != 'text_only' for finding in audit['issues']):
        return None
    rejected = {(row['candidate_id'], row['field']) for row in audit['field_reviews'] if not row['supported']}
    approved = {(row['candidate_id'], row['field']) for row in audit['field_reviews'] if row['supported']}
    edits = {}
    for finding in audit['issues']:
        owner, field = finding['candidate_id'], finding.get('field')
        key = (owner, field)
        text = finding.get('replacement')
        if (key not in rejected or key in approved or not isinstance(text, str) or not text.strip()
                or len(text) > (900 if field == 'summary' else 500 if field == 'size_reason' else 450)
                or not finding.get('sources')):
            return None
        def references(name):
            result = set()
            for ref in finding.get(name) or []:
                candidate, path = ref['candidate_id'], expand_reference(ref['path'])
                fact_value(catalog[candidate], path)
                result.add((candidate, path))
            return result
        try:
            added, removed = references('sources'), references('superseded_sources')
        except (KeyError, ValueError):
            return None
        edit = (text.strip(), added, removed)
        if added & removed or (key in edits and edits[key] != edit):
            return None
        edits[key] = edit
    if set(edits) != rejected:
        return None
    result = deepcopy(draft)
    by_id = {item['candidate_id']: item for item in result['recommendations']}
    additions, removals = {}, {}
    for (owner, field), (text, added, removed) in edits.items():
        if owner == 'panel':
            # Panel text has no separate citation field. Its evidence must
            # already be cited by a member of the panel.
            cited = {(key, p) for key, item in by_id.items() for p in item['evidence_paths']}
            cited.update((ref['candidate_id'], ref['path']) for item in by_id.values() for ref in item['comparison_evidence'])
            if field not in ('summary', 'size_reason') or removed or not added <= cited:
                return None
            result[field] = text
        else:
            if owner not in by_id or field not in ('why_chosen', 'tradeoff', 'panel_role'):
                return None
            by_id[owner][field] = text
            additions.setdefault(owner, set()).update(added)
            removals.setdefault(owner, set()).update(removed)
    for key, added in additions.items():
        item, removed = by_id[key], removals[key]
        current = [(key, path) for path in item['evidence_paths']]
        current.extend((ref['candidate_id'], ref['path']) for ref in item['comparison_evidence'])
        if not removed <= set(current) or removed & added:
            return None
        updated = dict.fromkeys([*(ref for ref in current if ref not in removed), *sorted(added)])
        item['evidence_paths'] = [path for owner, path in updated if owner == key]
        item['comparison_evidence'] = [{'candidate_id': owner, 'path': path} for owner, path in updated if owner != key]
        if not item['evidence_paths']:
            return None
    return result if result != draft else None


def mismatch_citation_findings(draft, claims, catalog):
    """Check the unit-specific bin named by the auditor against the cited original source."""
    by_id = {item['candidate_id']: item for item in draft['recommendations']}
    issues = []
    for claim in claims:
        owner, field = claim['candidate_id'], claim['field']
        source, ctx, mismatch = claim['source_candidate_id'], claim['context'], claim['mismatches']
        item = by_id[owner]
        entry = catalog[source]
        prefix = f'/contexts/{ctx}/'
        counts = prefix + 'candidate_assessment/off_target/counts_by_mismatch'
        indexed = counts + '/' + str(mismatch)
        native = 'Off_target_' + ('exact' if mismatch == 0 else str(mismatch) + 'mm') + '_count'
        possible = [indexed, prefix + native, prefix + 'source_annotations/' + native]
        matching, observed = [], set()
        for path in possible:
            try:
                value = fact_value(entry, path)
            except KeyError:
                continue
            if type(value) in (int, float):
                observed.add(value)
                if value == claim['count']:
                    matching.append(path)
        if len(observed) > 1:
            matching = []
        refs = item['evidence_paths'] if source == owner else [
            ref['path'] for ref in item['comparison_evidence'] if ref['candidate_id'] == source]
        if any(path in refs for path in matching) or (indexed in matching and counts in refs):
            continue
        correct = [{'candidate_id': source, 'path': path} for path in matching[:1]]
        problem = (f"The {mismatch}-mismatch count of {claim['count']} needs the original citation for that "
                   f"bin and molecule in context {ctx}.") if correct else (
                   f"The original {mismatch}-mismatch records for context {ctx} do not support the stated count of {claim['count']}.")
        issues.append({'candidate_id': owner, 'field': field, 'problem': problem,
                       'selection_impact': 'text_only' if correct else 'uncertain',
                       'replacement': item[field] if correct else None, 'sources': correct, 'superseded_sources': []})
    return issues


def apply_mismatch_findings(audit, findings, expand_reference):
    """Merge verified bin citations into the reviewer's edits, also on receipt replay."""
    if not findings:
        return
    # Earlier validations can have attached several missing bin references to
    # one citation-only repair. Rebuild those exact repairs from the claims.
    # Independently supplied replacements, extra evidence and objections stay.
    verified_sources = {}
    for issue in findings:
        verified_sources.setdefault((issue['candidate_id'], issue['field']), []).extend(issue['sources'])
    def generated_repair(edit):
        if edit.get('selection_impact') != 'text_only' or not edit.get('sources'):
            return False
        known = verified_sources.get((edit['candidate_id'], edit.get('field')), [])
        return all(ref in known for ref in edit['sources']) and any(
            {**edit, 'sources': issue['sources']} == issue for issue in findings)
    audit['issues'] = [edit for edit in audit['issues'] if not generated_repair(edit)]
    for issue in findings:
        wording_edits = [edit for edit in audit['issues']
            if issue['selection_impact'] == edit.get('selection_impact') == 'text_only'
            and (edit['candidate_id'], edit.get('field')) == (issue['candidate_id'], issue['field'])
            and edit.get('replacement')]
        combined = [list({(ref['candidate_id'], expand_reference(ref['path'])): ref
            for ref in (edit.get('sources') or []) + issue['sources']}.values()) for edit in wording_edits]
        if wording_edits and all(len(refs) <= 8 for refs in combined):
            for edit, refs in zip(wording_edits, combined):
                edit['sources'] = refs
        elif issue not in audit['issues']:
            audit['issues'].append(dict(issue))
        for field in audit['field_reviews']:
            if (field['candidate_id'], field['field']) == (issue['candidate_id'], issue['field']):
                field.update(supported=False, evidence_statement=issue['problem'])
