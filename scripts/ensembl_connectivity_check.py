#!/usr/bin/env python3
"""
Ensembl REST connectivity + GRCh38 gene lookup checker.

What this script guarantees:
- DNS resolution works (optionally IPv4-only)
- The main lookup endpoint is reachable
- Redirects are allowed for archive aliases
- The final effective host must be rest.ensembl.org
- The returned gene must be GRCh38 (assembly_name == "GRCh38")

This intentionally treats /lookup as the primary health signal.
Ping is informational only unless --strict-ping is set.
"""

import argparse
import json
import socket
import sys
import textwrap
from urllib.parse import urlparse

import requests


DEFAULT_GENE = "OTUD4"
DEFAULT_SPECIES = "homo_sapiens"
DEFAULT_TRANSCRIPT = ""

PRIMARY_HOST = "rest.ensembl.org"


def dns_lookup(host: str, ipv4_only: bool = True):
    """
    Returns (ok, info_str, addrs_list)
    """
    family = socket.AF_INET if ipv4_only else socket.AF_UNSPEC
    try:
        res = socket.getaddrinfo(host, 443, family, socket.SOCK_STREAM)
        addrs = sorted({r[4][0] for r in res})
        return True, ",".join(addrs) if addrs else "no addresses", addrs
    except Exception as e:
        return False, repr(e), []


def curl_like_line(title, ok, info):
    print(f"\n== {title} ==")
    print(f"{'ok' if ok else 'fail'}={ok} info={info}")


def http_get_json(url: str, headers: dict, timeout: float, allow_redirects: bool = True):
    """
    Return response, parsed_json_or_none, error_str_or_none
    """
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
        # Try JSON only when content-type hints or when status is 200-ish
        data = None
        try:
            data = r.json()
        except Exception:
            data = None
        return r, data, None
    except Exception as e:
        return None, None, repr(e)


def summarize_redirects(r: requests.Response):
    if r is None:
        return "no response"
    chain = []
    for h in r.history:
        chain.append(f"{h.status_code}:{h.url}")
    chain.append(f"{r.status_code}:{r.url}")
    return " -> ".join(chain)


def enforce_final_host(r: requests.Response, expected_host: str):
    final_host = urlparse(r.url).netloc if r is not None else ""
    if final_host != expected_host:
        raise RuntimeError(f"Unexpected final host '{final_host}', expected '{expected_host}'.")


def enforce_grch38(data: dict):
    assembly = (data or {}).get("assembly_name")
    if assembly != "GRCh38":
        raise RuntimeError(f"Unexpected assembly_name '{assembly}', expected 'GRCh38'.")


def build_lookup_url(host: str, species: str, gene: str, expand: int):
    return f"https://{host}/lookup/symbol/{species}/{gene}?expand={expand}"


def build_transcript_lookup_url(host: str, transcript_id: str, expand: int):
    return f"https://{host}/lookup/id/{transcript_id}?expand={expand}"


def build_ping_url(host: str):
    return f"https://{host}/info/ping"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            f"""
            Ensembl main REST API GRCh38 health check.

            Default behavior:
            - DNS check for rest.ensembl.org (IPv4)
            - Optional alias DNS checks if you pass --alias
            - Ping is informational
            - Lookup MUST succeed and MUST be GRCh38

            Examples:
              python scripts/ensembl_connectivity_check.py
              python scripts/ensembl_connectivity_check.py --gene TP53
              python scripts/ensembl_connectivity_check.py --alias e115.rest.ensembl.org
              python scripts/ensembl_connectivity_check.py --alias sep2025.rest.ensembl.org --strict-ping
            """
        ),
    )

    ap.add_argument("--gene", default=DEFAULT_GENE)
    ap.add_argument("--species", default=DEFAULT_SPECIES)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--expand", type=int, default=1, choices=[0, 1])
    ap.add_argument("--transcript", default=DEFAULT_TRANSCRIPT, help="Transcript ID to lookup; defaults to first transcript from the gene lookup when expand=1.")
    ap.add_argument("--ipv4-only", action="store_true", default=True)
    ap.add_argument("--no-ipv4-only", action="store_false", dest="ipv4_only")

    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Optional archive/alias hostnames to test (can repeat).",
    )

    ap.add_argument(
        "--strict-ping",
        action="store_true",
        help="Fail the script if ping is not HTTP 200 on the final host.",
    )

    ap.add_argument(
        "--print-json",
        action="store_true",
        help="Print the lookup JSON body (can be large with expand=1).",
    )

    args = ap.parse_args()

    headers = {
        "Accept": "application/json",
        "User-Agent": "MODAL-ASO-ensembl-connectivity-check",
    }

    overall_ok = True

    # 1) DNS for primary
    ok, info, _ = dns_lookup(PRIMARY_HOST, ipv4_only=args.ipv4_only)
    curl_like_line(f"DNS lookup ({PRIMARY_HOST})", ok, info)
    overall_ok &= ok

    # 2) DNS for aliases
    for host in args.alias:
        ok, info, _ = dns_lookup(host, ipv4_only=args.ipv4_only)
        curl_like_line(f"DNS lookup ({host})", ok, info)
        overall_ok &= ok

    # 3) Ping primary (informational unless strict)
    ping_url = build_ping_url(PRIMARY_HOST)
    r, data, err = http_get_json(ping_url, headers=headers, timeout=args.timeout, allow_redirects=True)
    if err:
        curl_like_line(f"GET ping (primary)", False, err)
        if args.strict_ping:
            overall_ok = False
    else:
        info = f"status={r.status_code}, chain={summarize_redirects(r)}"
        ok = (r.status_code == 200)
        # Don't enforce JSON shape hard; ping sometimes returns {"ping":1}
        curl_like_line("GET ping (primary)", ok, info)
        if args.strict_ping:
            overall_ok &= ok

    # 4) Lookup primary - this is the real gate
    selected_transcript: str = (args.transcript or "").strip()
    lookup_url = build_lookup_url(PRIMARY_HOST, args.species, args.gene, args.expand)
    r, data, err = http_get_json(lookup_url, headers=headers, timeout=args.timeout, allow_redirects=True)
    if err:
        curl_like_line("GET lookup (primary)", False, err)
        overall_ok = False
    else:
        try:
            enforce_final_host(r, PRIMARY_HOST)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            if not isinstance(data, dict):
                raise RuntimeError("Response was not JSON object.")
            enforce_grch38(data)
            info = f"status=200, assembly={data.get('assembly_name')}, effective={r.url}"
            curl_like_line("GET lookup (primary)", True, info)
            if args.expand == 1 and not selected_transcript:
                transcripts = data.get("Transcript") or []
                # Prefer canonical/MANE if available, else first
                def _pick_tx(entries):
                    canonical = next((t for t in entries if t.get("is_canonical")), None)
                    if canonical and canonical.get("id"):
                        return canonical["id"]
                    mane = next((t for t in entries if t.get("is_mane_select") or t.get("mane_select") or t.get("MANE_Select")), None)
                    if mane and mane.get("id"):
                        return mane["id"]
                    first = next((t for t in entries if t.get("id")), None)
                    return first["id"] if first else ""
                selected_transcript = _pick_tx(transcripts)
        except Exception as e:
            info = f"status={r.status_code}, chain={summarize_redirects(r)}, err={e}"
            curl_like_line("GET lookup (primary)", False, info)
            overall_ok = False

        if args.print_json and isinstance(data, dict):
            print("\n--- Lookup JSON ---")
            print(json.dumps(data, indent=2, sort_keys=False))

    # 4b) Transcript lookup using the same API; this must succeed
    if selected_transcript:
        tx_url = build_transcript_lookup_url(PRIMARY_HOST, selected_transcript, args.expand)
        r_tx, data_tx, err_tx = http_get_json(tx_url, headers=headers, timeout=args.timeout, allow_redirects=True)
        title = f"GET transcript lookup ({selected_transcript})"
        if err_tx:
            curl_like_line(title, False, err_tx)
            overall_ok = False
        else:
            try:
                enforce_final_host(r_tx, PRIMARY_HOST)
                if r_tx.status_code != 200:
                    raise RuntimeError(f"HTTP {r_tx.status_code}")
                if not isinstance(data_tx, dict):
                    raise RuntimeError("Response was not JSON object.")
                enforce_grch38(data_tx)
                info = f"status=200, assembly={data_tx.get('assembly_name')}, effective={r_tx.url}"
                curl_like_line(title, True, info)
            except Exception as e:
                info = f"status={r_tx.status_code}, chain={summarize_redirects(r_tx)}, err={e}"
                curl_like_line(title, False, info)
                overall_ok = False
    else:
        curl_like_line("GET transcript lookup (skipped)", False, "No transcript ID available (provide --transcript or run with expand=1).")
        overall_ok = False

    # 5) Lookup aliases - allow redirects but require final primary host + GRCh38
    for host in args.alias:
        alias_lookup_url = build_lookup_url(host, args.species, args.gene, args.expand)
        r, data, err = http_get_json(alias_lookup_url, headers=headers, timeout=args.timeout, allow_redirects=True)
        title = f"GET lookup (alias {host})"

        if err:
            curl_like_line(title, False, err)
            overall_ok = False
            continue

        try:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            enforce_final_host(r, PRIMARY_HOST)
            if not isinstance(data, dict):
                raise RuntimeError("Response was not JSON object.")
            enforce_grch38(data)
            info = f"status=200, chain={summarize_redirects(r)}, effective={r.url}"
            curl_like_line(title, True, info)
        except Exception as e:
            info = f"status={r.status_code}, chain={summarize_redirects(r)}, err={e}"
            curl_like_line(title, False, info)
            overall_ok = False

    print("\n== Summary ==")
    print("overall_ok=", overall_ok)

    sys.exit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
