from copy import deepcopy

import pytest

from aso_agents.pipeline_tools import (
    _derive_exploratory_steric_programs,
    _program_isoform_identity_aliases,
)


STABLE = "ENST00000000001"
VERSIONED = STABLE + ".8"


def goal():
    return {
        "intent": "splice_modulation",
        "isoform_transcripts": {
            STABLE: {
                "transcript_id": STABLE,
                "metadata": {
                    "id": STABLE,
                    "versioned_id": VERSIONED,
                    "version": 8,
                    "reference_source": "local_gtf",
                },
            }
        },
        "splice_site_plan": [
            {"chrom": "chr1", "position": position, "site": site,
             "strand": "+", "delta_direction": "decrease", "present_in": [STABLE],
             "exon_indices": {STABLE: 3}, "delta_per_isoform": {STABLE: "decrease"}}
            for site, position in (("acceptor", 1000), ("donor", 1100))
        ],
    }


def program(transcript=VERSIONED):
    return {"program_id": "skip_exon", "modality": "steric", "intent": "splice_modulation",
            "transcript_id": transcript, "exon_label": 3, "site_kind": "both",
            "desired_delta": "decrease", "desired_effect": "skipping",
            "targeting_policy": "default_exploratory", "run_splice_eval": True}


@pytest.mark.parametrize("version", [8, "8", None])
def test_exact_attested_version_resolves_stable_splice_evidence(version):
    data = goal()
    data["isoform_transcripts"][STABLE]["metadata"]["version"] = version
    before = deepcopy(data)
    aliases = _program_isoform_identity_aliases(data, program())
    assert aliases == {STABLE.casefold(), VERSIONED.casefold()}
    derived, evidence = _derive_exploratory_steric_programs(structured_goal=data, base_program=program())
    assert len(derived) == 1
    assert evidence["site_labels"] == ["acceptor@1000", "donor@1100"]
    assert data == before


@pytest.mark.parametrize("requested", [STABLE + ".7", STABLE + ".9", "ENST00000000002.8"])
def test_unattested_requested_version_or_identity_does_not_resolve(requested):
    data = goal()
    aliases = _program_isoform_identity_aliases(data, program(requested))
    assert STABLE.casefold() not in aliases
    derived, evidence = _derive_exploratory_steric_programs(structured_goal=data, base_program=program(requested))
    assert derived == []
    assert evidence["status"] == "unresolved_named_route_scope"


@pytest.mark.parametrize("field,value", [
    ("versioned_id", "ENST00000000002.8"),
    ("versioned_id", STABLE + ".unsupported"),
    ("versioned_id", ""),
    ("id", "ENST00000000002"),
    ("id", STABLE + ".7"),
    ("version", 7),
])
def test_inconsistent_reference_metadata_cannot_create_alias(field, value):
    data = goal()
    data["isoform_transcripts"][STABLE]["metadata"][field] = value
    assert _program_isoform_identity_aliases(data, program()) == {VERSIONED.casefold()}


def test_explicit_resolved_version_conflict_cannot_create_alias():
    data = goal()
    data["isoform_transcripts"][STABLE]["transcript_id"] = STABLE + ".7"
    assert _program_isoform_identity_aliases(data, program()) == {VERSIONED.casefold()}


def test_named_isoform_keeps_its_identity_when_version_alias_is_added():
    data = goal()
    data["isoform_transcripts"]["included isoform"] = data["isoform_transcripts"].pop(STABLE)
    assert _program_isoform_identity_aliases(data, program()) == {
        "included isoform", STABLE.casefold(), VERSIONED.casefold(),
    }


def test_alias_bridge_does_not_override_opposing_route_direction():
    data = goal()
    route = program()
    route["desired_delta"] = "increase"
    derived, evidence = _derive_exploratory_steric_programs(structured_goal=data, base_program=route)
    assert derived == []
    assert evidence["status"] == "unresolved_named_route_scope"
