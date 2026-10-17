"""Typed, evidence-bound reference decisions; never interpret intent with keywords."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from agents import Agent
from aso_agents.async_utils import run_async_from_sync
from aso_agents.llm_provider import provider_credentials_available


def semantic_result(name, instructions, payload, schema, validator=None):
    # Import lazily: the pipeline owns the existing schema-repair/model policy.
    from aso_agents import pipeline_tools as pipeline

    async def invoke():
        base = pipeline.request_workbench_agent
        agent = pipeline._agent_with_strict_output_schema(
            Agent(name=name, instructions=instructions, tools=[], model=base.model,
                  model_settings=base.model_settings),
            schema_name=name.lower().replace(" ", "_"), schema=schema,
        )
        return await pipeline._run_schema_validated_agent_once(
            agent=agent, prompt=json.dumps(payload, ensure_ascii=False), schema=schema,
            error_prefix=name.lower().replace(" ", "_"), contract_validator=validator,
        )
    if not provider_credentials_available():
        return {"error": "semantic_reference_provider_unavailable"}
    return run_async_from_sync(invoke)


def provisioning_decision(state, preflight):
    clarification = state.get("clarification") or {}
    if clarification.get("kind") != "reference_preflight":
        return {"action": "ask"}
    answers = [dict(item) for item in clarification.get("last_answers", [])
               if isinstance(item, dict) and str(item.get("answer") or "").strip()]
    if not answers:
        return {"action": "ask"}
    payload = {"answers": answers, "preflight": preflight}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = state.get("reference_provisioning_resolution") or {}
    if cached.get("input_sha256") == fingerprint:
        return deepcopy(cached["decision"])
    properties = {
        "action": {"type": "string", "enum": ["ask", "continue", "use_existing", "download", "install_splice", "skip", "stop"]},
        "scope": {"type": "string", "enum": ["task", "species", "resources", "none"]},
        "species": {"type": "array", "items": {"type": "string"}},
        "answer_index": {"type": ["integer", "null"]},
        "evidence": {"type": "string"},
        "reference_data_root": {"type": "string"},
        "download_dir": {"type": "string"},
        "model_path": {"type": "string"},
        "species_resolutions": {"type": "array", "items": {"type": "object", "properties": {
            "requested": {"type": "string"}, "reference_name": {"type": "string"}},
            "required": ["requested", "reference_name"], "additionalProperties": False}},
        "reason": {"type": "string"},
    }
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    def validate(result):
        errors = []
        index = result.get("answer_index")
        if result.get("action") != "ask":
            if type(index) is not int or not 0 <= index < len(answers):
                return ["An action needs a supplied answer index."]
            answer = str(answers[index].get("answer") or "")
            if result.get("evidence") != answer:
                errors.append("Preserve the complete authorizing answer as evidence, including its scope and negations.")
            for key in ("reference_data_root", "download_dir", "model_path"):
                if result.get(key) and result[key] not in answer:
                    errors.append(f"{key} must be a literal path supplied in the answer.")
            for resolution in result.get("species_resolutions", []):
                if resolution.get("requested") not in (preflight.get("unresolved_species") or {}):
                    errors.append("Species corrections must address an unresolved requested species.")
                if not resolution.get("reference_name") or resolution["reference_name"] not in answer:
                    errors.append("A reference name must be supplied literally in the answer.")
        return errors

    result = semantic_result(
        "Reference Provisioning Decision",
        "Interpret the supplied reference clarification answers semantically. Select an action only when "
        "the complete answer authorizes it. Respect negation, conditionals and scope. Retaining a species "
        "or supplying an identifier is continue; declining a download does not authorize skipping a design. "
        "Skip or stop requires explicit affirmative authorization at the indicated scope. Do not infer "
        "authorization from questions or suggested answers. Use ask for ambiguity. Extract only literal "
        "supplied paths. Record supplied organism identity corrections in species_resolutions without "
        "dropping species. For an action copy the entire authorizing answer into evidence. Return the schema.",
        payload, schema, validate,
    )
    if result.get("error") or validate(result):
        decision = {"action": "ask", "error": result.get("error") or "invalid_reference_decision"}
    else:
        decision = dict(result)
        decision["answer"] = "\n".join(str(item.get("answer") or "") for item in answers)
        if decision["action"] == "use_existing" and not decision.get("reference_data_root"):
            decision["action"] = "continue"
            decision["resource_effect"] = "recheck_current_reference_configuration"
        if decision["action"] == "skip" and decision["scope"] != "task":
            # Never expand a species-specific cancellation to the whole request.
            decision.update(action="ask", reason="A partial-species skip requires a scoped goal revision.")
    state["reference_provisioning_resolution"] = {"input_sha256": fingerprint, "decision": deepcopy(decision)}
    return decision


def apply_species_resolutions(goal, resolutions):
    from aso_agents.species_config import resolve_ensembl_species, normalize_species
    replacements = {}
    evidence = []
    for correction in resolutions:
        requested = normalize_species(correction["requested"])
        resolved = resolve_ensembl_species(correction["reference_name"])
        if resolved.get("status") != "resolved":
            continue
        replacements[requested] = resolved["species_key"]
        evidence.append({"requested": correction["requested"], "resolution": resolved})
    species_fields = {"species", "species_pair", "species_set", "organisms", "reference_species",
                      "match_species", "target_species", "species_required_match"}
    def visit(value):
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key in species_fields:
                    if isinstance(child, str):
                        value[key] = replacements.get(normalize_species(child), child)
                    elif isinstance(child, list):
                        value[key] = [replacements.get(normalize_species(item), item) if isinstance(item, str) else item for item in child]
                elif key != "species_reference_resolutions":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(goal)
    if evidence:
        recorded = goal.setdefault("species_reference_resolutions", [])
        recorded.extend(item for item in evidence if item not in recorded)
    return evidence


def resolve_gene_label(label, species, lookup):
    """Extract literal names from a composite label, then require database agreement."""
    schema = {"type": "object", "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
        "single_gene": {"type": "boolean"},
    }, "required": ["names", "single_gene"], "additionalProperties": False}
    result = semantic_result(
        "Gene Label Names",
        "Extract the literal gene symbol/name/alias components explicitly present in this label. "
        "Do not propose identifiers or names absent from the label. Indicate whether it describes "
        "one gene with aliases or multiple/different genes. Return the schema.",
        {"label": label, "species": species}, schema,
    )
    names = list(dict.fromkeys(result.get("names") or []))
    if result.get("single_gene") is not True or not names or any(name not in label for name in names):
        return {"error": "gene_label_identity_unresolved", "interpretation": result}
    candidates = [lookup(name, species=species) for name in names]
    resolved = [item for item in candidates if item.get("gene_id") and not item.get("error")]
    ids = {item["gene_id"] for item in resolved}
    if len(ids) != 1:
        return {"error": "gene_label_identity_ambiguous_or_unresolved", "candidates": candidates}
    selected = dict(resolved[0])
    selected.update(resolution_source="reference_validated_label_names", label_input=label,
                    label_resolution={"interpretation": result, "lookups": candidates})
    return selected
