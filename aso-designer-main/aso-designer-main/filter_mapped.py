#!/usr/bin/env python
# encoding: utf-8
'''
Based on the Globus tutorial provided here:
https://globus-sdk-python.readthedocs.io/en/stable/examples/minimal_transfer_script/index.html#example-minimal-transfer

@author:     Brian Yee

@copyright:  2024 Yeolab. All rights reserved.

@license:    license

@contact:    bay001@ucsd.edu
@deffield    updated: 2024-02-13
'''
import sys
import os
import glob
import pybedtools
import pandas as pd
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pytz import timezone, utc
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
pd.options.mode.chained_assignment = None

__all__ = []
__version__ = '1.0.0'
__date__ = '2024-02-13'
__updated__ = '2024-02-13'

utc_time = datetime.utcnow()
pst_tz = timezone('US/Pacific')
pst_time = utc_time.replace(tzinfo=utc).astimezone(pst_tz)
today = pst_time.strftime("%Y%m%d")


def fix_strand(row):
    return str(Seq(row['seq']).reverse_complement()) if row['flag'] == 16 else row['seq']


def format_position(row, tile_len=20):
    return "{}:{}-{}".format(row['rname'], row['pos'], int(row['pos']) + tile_len)


def fix_strand_values(flag: int, seq: str) -> str:
    return str(Seq(seq).reverse_complement()) if flag & 16 else seq


def parse_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def extract_nm_tag(optional_fields: List[str]) -> int:
    for field in optional_fields:
        if field.startswith("NM:i:"):
            return parse_int(field.rsplit(":", 1)[-1], 0)
    return 0


def parse_query_target_start(qname: str) -> Tuple[Optional[str], Optional[int]]:
    for token in str(qname).split("|"):
        if not token.startswith("pos="):
            continue
        raw = token.split("=", 1)[1]
        parts = raw.split(":")
        if len(parts) < 2:
            continue
        chrom = parts[0]
        span = parts[1]
        if "-" not in span:
            continue
        start = parse_int(span.split("-", 1)[0], -1)
        if chrom and start >= 0:
            return chrom, start
    return None, None


def is_intended_target_position(qname: str, rname: str, pos: int) -> bool:
    target_chrom, target_start = parse_query_target_start(qname)
    return bool(target_chrom and target_start is not None and target_chrom == rname and target_start == pos)


def load_sam_alignments(sam_path: str, noalt: bool = False, norandom: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(sam_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("@"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            qname = parts[0]
            flag = parse_int(parts[1], 0)
            rname = parts[2]
            pos = parse_int(parts[3], 0)
            mapq = parse_int(parts[4], 0)
            seq = parts[9]
            # MAPQ=0 still records a real placement (often a repetitive oligo).
            # Discarding it turns the highest mapping burdens into false zeros.
            if flag & 4 or not rname or rname == "*" or pos < 1:
                continue
            if noalt and (rname.endswith("_alt") or rname.endswith("_fix")):
                continue
            if norandom and rname.endswith("_random"):
                continue
            qseq = fix_strand_values(flag, seq)
            end_pos = pos + len(qseq) - 1
            rows.append(
                {
                    "qname": qname,
                    "flag": flag,
                    "rname": rname,
                    "pos": pos,
                    "mapq": mapq,
                    "seq": seq,
                    "qseq": qseq,
                    "position": f"{rname}:{pos}-{end_pos}",
                    "mismatches": extract_nm_tag(parts[11:]),
                    "is_intended_target": is_intended_target_position(qname, rname, pos),
                }
            )
    return pd.DataFrame(rows)


def build_mismatch_metadata(
    near_df: pd.DataFrame,
    max_mismatches: int,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    bins = list(range(0, max(3, max_mismatches) + 1))
    grouped: Dict[Tuple[str, str], Dict[int, set]] = defaultdict(lambda: {mm: set() for mm in bins})
    if not near_df.empty:
        for _, row in near_df.iterrows():
            mm = parse_int(row.get("mismatches"), 0)
            if mm < 0 or mm > max_mismatches:
                continue
            key = (str(row.get("qname")), str(row.get("qseq")))
            grouped[key]  # A fully searched intended-only query has true zeros.
            if bool(row.get("is_intended_target")):
                continue
            grouped[key].setdefault(mm, set()).add(str(row.get("position")))

    metadata: Dict[Tuple[str, str], Dict[str, str]] = {}
    all_keys = set(grouped.keys())
    for key in all_keys:
        per_bin = grouped.get(key, {})
        values: Dict[str, str] = {}
        total = 0
        for mm in bins:
            positions = sorted(per_bin.get(mm, set()))
            count = len(positions)
            total += count if mm <= max_mismatches else 0
            label = "exact" if mm == 0 else f"{mm}mm"
            values[f"off_target_{label}_count"] = str(count) if mm <= max_mismatches else "NA"
            values[f"off_target_{mm}mm_positions"] = (",".join(positions) if positions else "None") if mm <= max_mismatches else "NA"
        values[f"off_target_total_upto_{max_mismatches}mm_count"] = str(total)
        values["off_target_mismatch_budget"] = str(max_mismatches)
        values["off_target_weighted_risk_score"] = str(
            len(per_bin.get(0, set())) * 100
            + len(per_bin.get(1, set())) * 10
            + len(per_bin.get(2, set())) * 3
            + len(per_bin.get(3, set()))
        )
        metadata[key] = values
    return metadata


def reconcile_mapping_metadata(exact_df, near_df, query_keys, max_mismatches,
                               *, near_available, near_overflow=(), exact_overflow=(), same_reference=True):
    """Keep censored/unsearched bins unknown, preserving independently known exact hits."""
    near = build_mismatch_metadata(near_df, max_mismatches)
    exact = build_mismatch_metadata(exact_df, 0) if same_reference else {}
    near_overflow, exact_overflow = set(near_overflow), set(exact_overflow)
    result = {}
    for key in query_keys:
        name, _sequence = key
        values = dict(near.get(key) or {})
        complete = near_available and name not in near_overflow and key in near
        if not complete:
            values = {f'off_target_{label}_count':'NA' for label in ('exact','1mm','2mm','3mm')}
            values.update({f'off_target_{mm}mm_positions':'NA' for mm in range(4)})
            # An exact-only search is independently complete even when the
            # higher-mismatch search suppressed this query at its hit cap.
            if same_reference and name not in exact_overflow and key in exact:
                values['off_target_exact_count'] = exact[key]['off_target_exact_count']
                values['off_target_0mm_positions'] = exact[key]['off_target_0mm_positions']
            values[f'off_target_total_upto_{max_mismatches}mm_count'] = 'NA'
            values['off_target_weighted_risk_score'] = 'NA'
            if max_mismatches == 0 and values['off_target_exact_count'] != 'NA':
                values['off_target_total_upto_0mm_count'] = values['off_target_exact_count']
                values['off_target_weighted_risk_score'] = str(int(values['off_target_exact_count']) * 100)
        values['off_target_mismatch_budget'] = str(max_mismatches)
        values['off_target_search_status'] = 'complete' if complete else 'incomplete_or_censored'
        values['off_target_exact_count_unit'] = 'loci'
        values['off_target_position_convention'] = '1based_inclusive'
        result[key] = values
    return result


def dataframe_to_fasta(df, output_file):
    """
    Convert a pandas DataFrame with sequence headers and sequences into a FASTA file.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    header_col (str): Name of column containing sequence headers
    sequence_col (str): Name of column containing sequences
    output_file (str): Path to output FASTA file
    """
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Iterate through the DataFrame rows
        for _, row in df.iterrows():
            # Write header line (starting with >)
            # Remove any existing '>' from header if present
            header = str(row['header']).strip('>')
            f.write(f'>{header}\n')
            
            # Write sequence line
            # Remove any whitespace from sequence
            sequence = str(row['qseq']).strip()
            f.write(f'{sequence}\n')


def main(argv=None):  # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = f"v{__version__}"
    program_build_date = str(__updated__)
    program_version_message = f"{program_version} ({program_build_date})"
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = f'''{program_shortdesc}

  Created by Brian Yee on {str(__date__)}.
  Copyright 2016 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.
'''

    # Setup argument parser
    parser = ArgumentParser(
        description=program_license,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-V', '--version',
        action='version',
        version=program_version_message
    )
    parser.add_argument(
        "-s", "--sam-tab",
        dest="samtab",
        help="SAM-like file (SAM file without headers)",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-f", "--secondary-feature-tab",
        dest="featuretab",
        help="featurecounts -R output",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-na", "--no-alt",
        dest="noalt",
        help="Do not count alignments to '*_alt' contigs (eg. chr8_KI270816v1_alt). Default: count all.",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-nr", "--no-random",
        dest="norandom",
        help="Do not count alignments to '*_random' contigs (eg. chr16_KI270728v1_random). Default: count all.",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-o", "--fasta-out",
        dest="outfile",
        help="output fasta file",
        required=False,
    )
    parser.add_argument(
        "-x", "--overflow",
        dest="overflow",
        help="FASTA file containing reads exceeding the Bowtie -m threshold (produced via --max).",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--near-sam",
        dest="near_sam",
        help="SAM file containing near-match alignments for mismatch-bin off-target annotation.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--max-mismatches",
        dest="max_mismatches",
        help="Maximum mismatch bin to report from the near-match SAM.",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument('--near-overflow', help='Queries suppressed by the near-match Bowtie hit cap.')
    parser.add_argument('--same-reference', action='store_true', help='Exact and near alignments use the same reference index.')
    
    print("Begin at:", datetime.now().strftime("%H:%M:%S"))
    args = parser.parse_args()
    outfile = args.outfile
    samtab = args.samtab
    featuretab = args.featuretab
    noalt = args.noalt
    norandom = args.norandom
    overflow_path = args.overflow
    near_sam = args.near_sam
    max_mismatches = max(0, min(int(args.max_mismatches), 3))

    df = load_sam_alignments(samtab, noalt=noalt, norandom=norandom)
    if df.empty:
        collapsed = pd.DataFrame(columns=["qname", "qseq", "position"])
    else:
        collapsed = df.groupby(['qname', 'qseq'], as_index=False).agg({
            'position': lambda x: sorted(set(x.astype(str)))
        })

    near_df = pd.DataFrame()
    if near_sam and os.path.exists(near_sam):
        near_df = load_sam_alignments(near_sam, noalt=noalt, norandom=norandom)

    collapsed['mappednum'] = collapsed['position'].apply(lambda positions: f"mappednum={len(positions)}")
    collapsed['position'] = collapsed['position'].apply(lambda positions: 'mappedto=' + ','.join(positions))
    sfeatures = pd.read_csv(featuretab, sep='\t', names=['qname', 'assigned', 'is_assigned', 'secondary_gene_name'])
    sfeatures = sfeatures[sfeatures['is_assigned']==1][['qname', 'secondary_gene_name']]
    sfeatures = sfeatures.groupby("qname", as_index=False).agg({
        "secondary_gene_name": lambda values: ",".join(sorted(set(str(v) for v in values if str(v) and str(v) != "nan")))
    })
    merged = pd.merge(collapsed, sfeatures, how='left', left_on=['qname'], right_on=['qname'])

    overflow_records = []
    if overflow_path and os.path.exists(overflow_path) and os.path.getsize(overflow_path) > 0:
        with open(overflow_path, "r", encoding="utf-8") as overflow_handle:
            for record in SeqIO.parse(overflow_handle, "fasta"):
                qname = record.id
                overflow_records.append({
                    "qname": qname,
                    "qseq": str(record.seq),
                    "mappednum": "mappednum=>2000",
                    "position": "mappedto=overflow",
                    "secondary_gene_name": "None",
                    "overflow_flag": "overflow=TRUE",
                })
    if overflow_records:
        overflow_df = pd.DataFrame(overflow_records)
        merged = pd.concat([merged, overflow_df], ignore_index=True, sort=False)

    if "overflow_flag" not in merged.columns:
        merged["overflow_flag"] = None

    merged.fillna("None", inplace=True)

    # With no mapped or overflow records, pandas returns a DataFrame from
    # row-wise apply rather than the Series expected by the header assignment.
    # An empty FASTA is the truthful downstream result for this input.
    if merged.empty:
        dataframe_to_fasta(merged, output_file=outfile)
        print("Finished at:", datetime.now().strftime("%H:%M:%S"))
        return 0

    def overflow_names(path):
        if not path or not os.path.isfile(path):
            return set()
        with open(path) as handle:
            return {record.id for record in SeqIO.parse(handle, 'fasta')}
    mismatch_metadata = reconcile_mapping_metadata(
        df, near_df, [(str(row.qname), str(row.qseq)) for row in merged.itertuples()], max_mismatches,
        near_available=bool(near_sam and os.path.isfile(near_sam)),
        near_overflow=overflow_names(args.near_overflow), exact_overflow=overflow_names(overflow_path),
        same_reference=args.same_reference)

    merged['header'] = merged['qname'] + '|' + merged["mappednum"] + '|' + merged["position"] + '|secondary_hit=' + merged['secondary_gene_name']
    def append_mismatch_metadata(row):
        key = (str(row.get("qname")), str(row.get("qseq")))
        values = mismatch_metadata[key]
        return str(row["header"]) + "|" + "|".join(f"{k}={v}" for k, v in values.items())

    merged['header'] = merged.apply(append_mismatch_metadata, axis=1)
    merged['header'] = merged.apply(
        lambda row: row['header'] + f"|{row['overflow_flag']}"
        if isinstance(row.get("overflow_flag"), str) and row["overflow_flag"] == "overflow=TRUE"
        else row['header'],
        axis=1
    )

    dataframe_to_fasta(merged, output_file=outfile)

    print("Finished at:", datetime.now().strftime("%H:%M:%S"))
if __name__ == "__main__":
    sys.exit(main())
