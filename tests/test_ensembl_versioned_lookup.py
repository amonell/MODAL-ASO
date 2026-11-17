"""Versioned lookups must recover from Ensembl's missing-ID HTTP 400."""

import importlib
import json

import pytest
import requests

chooser = importlib.import_module("aso_agents.transcript_chooser_agent")


def response(url, status, payload):
    result = requests.Response()
    result.url = url
    result.status_code = status
    result._content = json.dumps(payload).encode()
    return result


@pytest.fixture
def remote_only(monkeypatch):
    monkeypatch.delenv("ASO_DEPENDENCY_MODE", raising=False)
    monkeypatch.setattr(chooser, "_ENSEMBL_ID_LOOKUP_CACHE", {})
    monkeypatch.setattr(chooser, "_lookup_local_gtf_id", lambda *_: None)


@pytest.mark.parametrize("base", ["ENST00000376285", "ENSMUST00000018411"])
@pytest.mark.parametrize("version", [6, 7, None])
def test_missing_versioned_id_retries_stable_id_and_checks_version(
    remote_only, monkeypatch, base, version,
):
    requested = f"{base}.6"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/" + requested):
            return response(url, 400, {"error": f"ID '{requested}' not found"})
        assert url.endswith("/" + base)
        return response(url, 200, {
            "id": base, "version": version, "seq_region_name": "13",
            "start": 100, "end": 399, "strand": 1,
            "Exon": [{"start": 100, "end": 199}, {"start": 300, "end": 399}],
        })

    monkeypatch.setattr(chooser.requests, "get", get)
    result = chooser.get_transcript_structure(requested)

    assert calls == [f"{chooser.ENSEMBL_BASE_URL}/lookup/id/{requested}",
                     f"{chooser.ENSEMBL_BASE_URL}/lookup/id/{base}"]
    if version == 6:
        assert result["versioned_id"] == requested
        assert result["exon_count"] == 2
        assert "error" not in result
    else:
        assert "error" in result
        assert result["requested_transcript_id"] == requested
        assert "exons_tx" not in result


def test_missing_stable_and_versioned_ids_return_missing_transcript(remote_only, monkeypatch):
    def get(url, **kwargs):
        return response(url, 400, {"error": f"ID '{url.rsplit('/', 1)[1]}' not found"})

    monkeypatch.setattr(chooser.requests, "get", get)
    result = chooser.get_transcript_structure("ENST00000376285.6")
    assert "not found" in result["error"]
    assert "exons_tx" not in result


@pytest.mark.parametrize("path,payload", [
    ("/lookup/id/ENST00000376285.6", {"error": "invalid request scope"}),
    ("/lookup/id/ENST00000376285.6", {"error": "ID 'different-id' not found"}),
    ("/lookup/id/ENST00000376285.6", ["ID 'ENST00000376285.6' not found"]),
    ("/info/ping", {"error": "ID 'ENST00000376285.6' not found"}),
])
def test_other_http_400_errors_remain_errors(remote_only, monkeypatch, path, payload):
    monkeypatch.setattr(chooser.requests, "get", lambda url, **kw: response(url, 400, payload))
    with pytest.raises(chooser.EnsemblServiceError) as caught:
        chooser._get_ensembl_json(path)
    assert caught.value.status_code == 400
    assert caught.value.retryable is False


def test_non_json_http_400_remains_service_error(remote_only, monkeypatch):
    def get(url, **kwargs):
        result = response(url, 400, None)
        result._content = b"Bad request"
        return result

    monkeypatch.setattr(chooser.requests, "get", get)
    with pytest.raises(chooser.EnsemblServiceError) as caught:
        chooser._get_ensembl_json("/lookup/id/ENST00000376285.6")
    assert caught.value.status_code == 400
