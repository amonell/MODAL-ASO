"""Evidence-bound requirements, recorded before seeing chemistry capabilities."""
import json
from agents import Agent
from aso_chemistry import chemistry_label, chemistry_components, require_linkage_specification

FIELDS = ("platform", "sugar", "backbone", "wing_5prime", "dna_gap", "wing_3prime", "length",
          "linkages", "cytosine_methylation", "stereochemistry", "terminal_modifications", "conjugates", "other")

chemical_requirement_agent = Agent(
    name="ASO Chemical Requirement Recorder",
    instructions=(
        "Record explicit physical chemical requirements from the supplied source texts. "
        "Do not select, recommend, default, or decide whether chemistry is supported. "
        "Record all stated sugars, bases/base modifications, segment counts, total length, bonds, "
        "stereochemistry, terminal groups and attached conjugates, regardless of their biological role. "
        "An attached delivery group remains part of molecular identity. A numerical wing-gap-wing partition "
        "does not by itself specify wing sugar in prose. Do not invent unstated requirements. "
        "Use canonical names 2MOE, 2OME, MORPHOLINO, PS, PO, PHOSPHORODIAMIDATE, gapmer, steric_blocker "
        "when those identities are stated; retain any other chemical identity verbatim. "
        "Cytosine methylation can be all, none, or a verbatim different policy; OMe C is not 5-methylcytosine. "
        "Record each constraint with its source_id, exact supporting quote, operator equals/one_of/excludes "
        "and a typed value. The output is a fixed object with one nullable entry per named field. "
        "Use null for unstated fields. Wing chemistry and gap chemistry are distinct fields. "
        "Wing counts describe explicitly specified segments of a gapmer partition, not the total "
        "oligo length or the number of modified residues. A fully modified steric chain is not one "
        "long gapmer wing. Record its total residue count as length and an explicit absence of a "
        "DNA gap as dna_gap=0 and gap_chemistry=none; leave unstated wing counts null. Preserve "
        "any explicitly stated segment counts, including zeros, even if they are incompatible. "
        "Platform is functional gapmer/steric_blocker architecture, not a sugar name. "
        "Equals uses a scalar for sugars/counts/policies, arrays for "
        "position-specific linkages or attached groups. One_of uses a JSON array of alternative complete values; "
        "excludes uses an array of forbidden values. A requirement with no available field uses other, "
        "with its full description as value. Leave unstated fields null; do not erase a requirement because it "
        "sounds inconvenient. Return the supplied JSON schema only."
    ), tools=[], model="gpt-5-mini-2025-08-07")


def requirement_sources(goal):
    """Only descriptive chemistry fields need semantic recording; typed profiles are validated directly."""
    result = []
    for path, value in (("/chemistry", goal.get("chemistry")),
                        ("/constraints/chemistry", (goal.get("constraints") or {}).get("chemistry"))):
        if isinstance(value, str) and value.strip() and not chemistry_label(value):
            result.append({"source_id": path, "text": value})
    return result


FIELD_TYPES = {
    'platform': {'type':'string', 'enum':['gapmer','steric_blocker']},
    'wing_chemistry': {'type':'string', 'description':'Wing sugar or fully modified steric sugar. Use 2MOE, 2OME, MORPHOLINO for those stated identities; preserve other identities verbatim. DNA in a gap belongs in gap_chemistry.'},
    'gap_chemistry': {'type':'string', 'description':'Sugar in the central gap: DNA, none for explicitly absent gap, or verbatim different identity.'},
    'backbone': {'type':'string', 'description':'PS, PO, PHOSPHORODIAMIDATE for these identities, or a verbatim different backbone.'},
    'wing_5prime': {'type':'integer', 'minimum':0, 'description':'Explicit residue count in the 5prime segment of a wing-gap-wing partition. A fully modified chain does not state a wing count; its total count belongs in length.'},
    'dna_gap': {'type':'integer', 'minimum':0, 'description':'Count of DNA nucleotides in central gap. No DNA gap means zero.'},
    'wing_3prime': {'type':'integer', 'minimum':0, 'description':'Explicit residue count in the 3prime segment of a wing-gap-wing partition. Do not infer zero or another wing count from an unpartitioned fully modified chain.'},
    'length': {'type':'integer','minimum':1, 'description':'Total number of residues in the whole oligo, independently of whether a wing-gap-wing partition is specified.'},
    'linkages': {'type':'array', 'items':{'type':'string'}, 'description':'One 5prime-to-3prime PS/PO entry per internucleotide bond.'},
    'cytosine_methylation': {'type':'string','description':'all for all cytosines 5-methylated, none for all unmethylated, or a verbatim different policy. Never confuse OMe with base methylation.'},
    'stereochemistry': {'type':'string','description':'Explicit stereochemical requirement verbatim.'},
    'terminal_modifications': {'type':'array','items':{'type':'string'}},
    'conjugates': {'type':'array','items':{'type':'string'}, 'description':'All explicitly attached groups, including delivery conjugates; an unconjugated request is an empty array.'},
    'other': {'type':'string','description':'Any other physical chemical requirement, verbatim; exclude target biology and activity hypotheses.'}}


def requirement_schema(sources):
    fields = {}
    for field, kind in FIELD_TYPES.items():
        properties = {'operator':{'type':'string','enum':['equals','one_of','excludes']},
            'value': {'anyOf':[kind, {'type':'array','items':kind}]},
            'quote':{'type':'string'}, 'source_id':{'type':'string','enum':[s['source_id'] for s in sources]}}
        fields[field] = {'anyOf':[{'type':'null'}, {'type':'object','properties':properties,
                           'required':list(properties),'additionalProperties':False}]}
    return {'type':'object','properties':{'requirements':{'type':'object','properties':fields,
        'required':list(fields),'additionalProperties':False}},'required':['requirements'],'additionalProperties':False}


def validate_requirement_response(payload, sources):
    from jsonschema import validate
    validate(payload, requirement_schema(sources))
    texts={s['source_id']:s['text'] for s in sources}; requirements=[]
    for field, entry in payload['requirements'].items():
        if entry is None: continue
        if not entry['quote'] or entry['quote'] not in texts[entry['source_id']]:
            raise ValueError('Chemical requirement lacks an exact supporting source quote.')
        value=entry['value']
        validate(value, FIELD_TYPES[field] if entry['operator']=='equals' else {'type':'array','items':FIELD_TYPES[field]})
        aliases = {'wing_chemistry': {'MOE': '2MOE', 'OMe': '2OME', 'PMO': 'MORPHOLINO'},
                   'gap_chemistry': {'dna': 'DNA'},
                   'backbone': {'ps': 'PS', 'po': 'PO', 'phosphorodiamidate': 'PHOSPHORODIAMIDATE'},
                   'cytosine_methylation': {'unmethylated': 'none'},
                   'stereochemistry': {'unspecified': 'not_stereodefined', 'stereorandom': 'not_stereodefined'}}
        def normalized(v):
            return aliases.get(field, {}).get(v, v) if isinstance(v, str) else v
        value = normalized(value) if entry['operator'] == 'equals' else [normalized(v) for v in value]
        requirements.append({'field':'sugar' if field=='wing_chemistry' else field,
            'operator':entry['operator'],'value_json':json.dumps(value),'quote':entry['quote'],'source_id':entry['source_id']})
    return requirements


def _supported(field, value):
    if field == "sugar": return value in ("2MOE", "2OME", "MORPHOLINO")
    if field == "gap_chemistry": return value in ("DNA", "none")
    if field == "platform": return value in ("gapmer", "steric_blocker")
    if field == "backbone": return value in ("PS", "PHOSPHORODIAMIDATE")
    if field == "cytosine_methylation": return value in ("all", "none", "profile_default")
    if field == "stereochemistry": return value in (None, "not_stereodefined")
    if field in ("terminal_modifications", "conjugates"): return value == []
    if field == "linkages": return isinstance(value, list) and all(v == "PS" for v in value)
    if field in ("wing_5prime", "wing_3prime", "dna_gap", "length"):
        return type(value) is int and value >= (1 if field == "length" else 0) and (field != "dna_gap" or value == 0 or value >= 6)
    return False

def _constraint_matches(r, actual):
    value = json.loads(r['value_json'])
    if r['field'] == 'cytosine_methylation' and value == 'profile_default':
        return True
    if r['field'] == 'stereochemistry' and r['operator'] == 'excludes' and value and actual == 'not_stereodefined':
        return False
    return actual == value if r['operator'] == 'equals' else actual in value if r['operator'] == 'one_of' else actual not in value


def _require_compatible_building_blocks(requirements):
    from aso_chemistry import STERIC_SUGARS
    components = [chemistry_components(label) for label in STERIC_SUGARS]
    components.extend({'platform':'gapmer', 'wing_chemistry':sugar, 'gap_chemistry':'DNA', 'backbone':'PS'}
                      for sugar in set(STERIC_SUGARS.values()) if sugar != 'MORPHOLINO')
    fields = {'platform','sugar','gap_chemistry','backbone','cytosine_methylation',
              'stereochemistry','terminal_modifications','conjugates'}
    for c in components:
        represented = {'platform':c['platform'], 'sugar':c['wing_chemistry'], 'gap_chemistry':c['gap_chemistry'],
            'backbone':c['backbone'], 'cytosine_methylation':'all' if c['wing_chemistry']=='2MOE' else 'none' if c['wing_chemistry']=='2OME' else None,
            'stereochemistry':'not_stereodefined' if c['backbone']=='PS' else None,
            'terminal_modifications':[], 'conjugates':[]}
        if all(_constraint_matches(r, represented[r['field']]) for r in requirements if r['field'] in fields):
            return
    raise ValueError('The recorded combination of sugars, base modifications, backbone or attachments cannot be represented by native chemistry; do not substitute. Requirements: ' + '; '.join(r['quote'] for r in requirements))


def require_native_requirement_capabilities(requirements):
    for r in requirements:
        value = json.loads(r["value_json"])
        if r["operator"] == "excludes":
            continue
        options = value if r["operator"] == "one_of" else [value]
        if not any(_supported(r["field"], option) for option in options):
            raise ValueError(f"Required {r['field']} cannot be represented by native chemistry: {r['quote']}; do not substitute.")

    _require_compatible_building_blocks(requirements)


def requirement_configuration_errors(requirements, recommendation):
    """Compare the actual represented molecule against every recorded constraint."""
    label = recommendation["CHEMISTRY"]
    components = chemistry_components(label)
    from aso_chemistry import gapmer_architecture
    partition = gapmer_architecture(label)
    represented = {"platform": components["platform"], "sugar": components.get("wing_chemistry"),
        "backbone": components.get("backbone"), "gap_chemistry": components.get("gap_chemistry"), "length": recommendation["length"],
        "wing_5prime": partition.wing_5prime if partition else 0,
        "dna_gap": partition.dna_gap if partition else 0,
        "wing_3prime": partition.wing_3prime if partition else 0,
        "cytosine_methylation": "all" if components.get("wing_chemistry") == "2MOE" else "none" if components.get("wing_chemistry") == "2OME" else None,
        "stereochemistry": "not_stereodefined" if components.get("backbone") == "PS" else None, "terminal_modifications": [], "conjugates": [],
        "linkages": require_linkage_specification(recommendation["length"], recommendation.get("linkage_specification")) if components.get("backbone") == "PS" else None}
    errors = []
    for r in requirements:
        value = json.loads(r["value_json"]); actual = represented.get(r["field"])
        if r["field"] == "cytosine_methylation" and value == "profile_default":
            continue
        matches = actual == value if r["operator"] == "equals" else actual in value if r["operator"] == "one_of" else actual not in value
        if r["field"] == "stereochemistry" and r["operator"] == "excludes" and value and actual == "not_stereodefined":
            matches = False  # Undefined PS configuration cannot establish an excluded stereoisomer.
        if r["field"] == "other" or not matches:
            errors.append(f"Selected molecule does not fulfill recorded {r['field']}: {r['quote']}; do not discard the requirement.")
    return errors
