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
import math
import json
import pybedtools
import pandas as pd
import gzip
from collections import defaultdict
from Bio import SeqIO, bgzf
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction as GC
from datetime import datetime
from pytz import timezone, utc
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from upsetplot import plot
from matplotlib import pyplot as plt
from typing import Tuple
pd.options.mode.chained_assignment = None

__all__ = []
__version__ = '1.0.0'
__date__ = '2024-02-13'
__updated__ = '2024-02-13'

utc_time = datetime.utcnow()
pst_tz = timezone('US/Pacific')
pst_time = utc_time.replace(tzinfo=utc).astimezone(pst_tz)
today = pst_time.strftime("%Y%m%d")


def apply_configuration(aso, chemistry):
    # The CLI is also executed with the tiler directory as cwd. Import the
    # dependency-free shared molecular encoder from the enclosing application.
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from aso_chemistry import encode_idt_oligo
    return encode_idt_oligo(aso, chemistry)


def apply_modification(n, mod_type='2MOEr', p=0, linker=''):
    if mod_type == '2MOEr':
        return f"/i2MOEr{n}/{linker}" if p == 0 else f"/{p}2MOEr{n}/{linker}"
    elif mod_type == 'iMe-dC':
        return f"/iMe-d{n}/{linker}" if p == 0 else f"/{p}iMe-d{n}/{linker}"
    else:
        f"Modification type not recognized! [{mod_type}]"
        raise


def parse_mappednum(value: str) -> Tuple[int, str]:
    """Return (numeric_value, display_string) for mappednum metadata."""
    if value is None:
        return 0, ""
    text = str(value).strip()
    if not text:
        return 0, ""
    display = text
    prefix = ""
    while text and text[0] in (">", "<", "=", "~", ">=", "<="):
        prefix += text[0]
        text = text[1:]
    text = text.strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        numeric = int(digits)
    else:
        numeric = 0
    if prefix.startswith(">") and digits:
        numeric += 1
    return numeric, display


def filter(fasta, output_bed, output_idt, homopolymers, gc_min, gc_max, phastcon, phylop, modification):
    seq_di = defaultdict(dict)  # stores filter results for upset plotting
    off_target_columns = [
        ("Off_target_exact_count", "off_target_exact_count"),
        ("Off_target_1mm_count", "off_target_1mm_count"),
        ("Off_target_2mm_count", "off_target_2mm_count"),
        ("Off_target_3mm_count", "off_target_3mm_count"),
        ("Off_target_total_upto_mismatch_count", None),
        ("Off_target_mismatch_budget", "off_target_mismatch_budget"),
        ("Off_target_weighted_risk_score", "off_target_weighted_risk_score"),
        ("Off_target_0mm_positions", "off_target_0mm_positions"),
        ("Off_target_1mm_positions", "off_target_1mm_positions"),
        ("Off_target_2mm_positions", "off_target_2mm_positions"),
        ("Off_target_3mm_positions", "off_target_3mm_positions"),
    ]
    with open(fasta, 'r') as handle, open(output_bed, 'w') as ob, open(output_idt, 'w') as oi:
        oi.write(
            "Chrom\tStart\tEnd\tStrand\tNo_homopolymer\tGC\tPhastcon\tPhyloP\tNo_secondary_structure\tTimes_mapped\t"
            + "\t".join(label for label, _ in off_target_columns)
            + "\tSecondary_hit\tIDT\tSynthesis\tSequence\tScore"
            + "\tHomopolymer_excluded_motifs\tHomopolymer_screen_status\tHomopolymer_sequence_role\n"
        )
        for record in SeqIO.parse(handle, "fasta"):

            # Parse metadata from the sequence header (| delimited)
            metadata = {}
            fields = record.description.split('|')
            new_description = []
            for field in fields:
                if '=' in field:
                    key, value = field.split('=', 1)
                else:
                    continue
                metadata[key] = value
                if key != 'id' and key != 'pos':
                    new_description.append(field)

            # Build new description based on metadata
            new_description.append(f"seq={str(record.seq)}")
            id = metadata['id']
            new_description.insert(0, f"orig_id={id}")
            chrom, pos, strand = metadata['pos'].split(':')
            start, end = pos.split('-')
            new_description = '|'.join(new_description)

            # Apply filters
            synthesis = str(record.seq.reverse_complement())
            no_homopolymer = not any(sub in synthesis.upper() for sub in homopolymers)
            within_gc_bounds = True if (float(metadata['gc']) >= gc_min and float(metadata['gc']) <= gc_max) else False
            pass_phastcon = True if float(metadata['pc']) >= phastcon else False
            pass_phylop = True if float(metadata['ph']) >= phylop else False
            seq_di[str(record.seq)]['phastcons_measured'] = math.isfinite(float(metadata['pc']))
            seq_di[str(record.seq)]['phylop_measured'] = math.isfinite(float(metadata['ph']))
            no_struct = True if metadata['nostruct'] == 'YES' else False
            mapped_num, mapped_num_display = parse_mappednum(metadata.get('mappednum'))
            if mapped_num_display == "":
                mapped_num_display = str(mapped_num)
            secondary_hit = True if metadata['secondary_hit'] != "None" else False
            
            # Score based on filter
            code = 1000
            for filter, label in zip([no_homopolymer, within_gc_bounds, pass_phastcon, pass_phylop, no_struct, secondary_hit], ['no_homopolymer', 'within_gc', 'pass_phastcon', 'pass_phylop', 'no_struct', 'secondary_hit']):
                if filter is False:
                    code -= 100
                seq_di[str(record.seq)][label] = filter

            seq_di[str(record.seq)]['uniquely_mapped'] = True if mapped_num == 1 else False

            extra_hits = max(mapped_num - 1, 0)
            off_target_penalty = min(extra_hits, 3) * 100
            code -= off_target_penalty

            mismatch_budget = metadata.get("off_target_mismatch_budget", "")
            total_key = f"off_target_total_upto_{mismatch_budget}mm_count" if mismatch_budget else ""
            off_target_values = []
            for _, key in off_target_columns:
                if key is None:
                    off_target_values.append(metadata.get(total_key, "NA"))
                else:
                    off_target_values.append(metadata.get(key, "NA"))
            
            sequence = str(record.seq)
            idt = apply_configuration(synthesis, modification)
            chrom_out = chrom if chrom.startswith('chr') else f"chr{chrom}"
            oi.write(
                f"{chrom_out}\t{int(start)-1}\t{end}\t{strand}\t{no_homopolymer}\t{metadata['gc']}\t{metadata['pc']}\t{metadata['ph']}\t{no_struct}\t{mapped_num_display}\t"
                + "\t".join(str(value) for value in off_target_values)
                + f"\t{metadata['secondary_hit']}\t{idt}\t{synthesis}\t{sequence}\t{code}"
                + "\t" + json.dumps(list(homopolymers), separators=(',', ':'))
                + "\t" + ('not_requested' if not homopolymers else 'passed' if no_homopolymer else 'failed')
                + "\tsynthesized_antisense_sequence\n"
            )
            ob.write(f"{chrom_out}\t{int(start)-1}\t{end}\t{new_description}\t{code}\t{strand}\n")
    return seq_di


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
    program_license = (__import__("__main__").__doc__ or "Filter sequences based on various criteria").split("\n")[0]

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
        "-f", "--fasta",
        dest="fasta",
        help="Fasta file containing tiles to be filtered.",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-p", "--homopolymers",
        dest="homopolymers",
        help="filter sequences including homopolymer(s) (default: none)",
        required=False,
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "-g", "--gc-min",
        dest="gc_min",
        help="filter sequences below GC content (default: 0.4)",
        required=False,
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "-G", "--gc-max",
        dest="gc_max",
        help="filter sequences above GC content (default: 0.6)",
        required=False,
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "-pt", "--phastcon-threshold",
        dest="phastcon",
        help="phastcon conservation threshold (Default: filter seqs w/ average phastcon score lower than 0.99)",
        required=False,
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-yt", "--phylop-threshold",
        dest="phylop",
        help="phylop conservation threshold (Default: filter seqs w/ average phastcon score lower than 0.99)",
        required=False,
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-ob", "--output-bed",
        dest="output",
        help="File to save tiles to (default: {fasta basename}.scored.bed",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-oi", "--output-idt",
        dest="outputi",
        help="IDT-formatted file to save tiles to (default: {fasta basename}.scored.idt.tsv",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ou", "--output-upset",
        dest="outputu",
        help="Upset plot representation of overlapping filters (default: {fasta basename}.upset.pdf)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-m", "--chemistry",
        dest="modification",
        help="Physical chemistry: numeric MOE/DNA wing-gap-wing (default: 5-10-5), 2ome:W-G-V, steric-hindrance (MOE/PS), steric-2ome-ps (OMe/PS), or steric-pmo (morpholino/phosphorodiamidate; no IDT code)",
        required=False,
        default='5-10-5',
    )
    
    print("Begin at:", datetime.now().strftime("%H:%M:%S"))
    args = parser.parse_args()
    fasta = args.fasta
    gc_min = args.gc_min
    gc_max = args.gc_max
    homopolymers = args.homopolymers
    phastcon = args.phastcon
    phylop = args.phylop
    modification = args.modification
    if fasta.endswith('.gz'):
        basename = os.path.splitext(os.path.splitext(fasta)[0])[0]
    else:
        basename = os.path.splitext(fasta)[0]
        
    output = basename + f".scored.bed" if args.output is None else args.output
    outputi = basename + f".scored.idt.tsv" if args.outputi is None else args.outputi
    outputu = basename + f".upset.pdf" if args.outputu is None else args.outputu
    
    records = defaultdict(list)
    values = filter(fasta, output, outputi, homopolymers, gc_min, gc_max, phastcon, phylop, modification)
    df = pd.DataFrame(values).T
    if df.empty:
        print("No sequences available for scoring; emitting empty summary.")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No tiles passed preprocessing\n(no alignments retained).", ha='center', va='center')
        ax.axis('off')
        fig.savefig(outputu)
        plt.close(fig)
        print("Finished at:", datetime.now().strftime("%H:%M:%S"))
        return

    plot_fields = ['within_gc', 'pass_phastcon', 'pass_phylop', 'no_struct', 'uniquely_mapped', 'secondary_hit']
    if homopolymers:
        plot_fields.insert(0, 'no_homopolymer')
    # A threshold pass requires a finite annotation. Show availability whenever
    # any annotation is missing so failure to establish a pass is not presented
    # as a measured low conservation value.
    for field in ('phastcons_measured', 'phylop_measured'):
        if not df[field].all():
            plot_fields.append(field)
    collapsed_df = df.groupby(plot_fields).size().reset_index(name='value')
    collapsed_df.set_index(plot_fields, inplace=True)
    
    plot(collapsed_df['value'], show_counts=True)
    import textwrap
    exclusions = ', '.join(homopolymers) if homopolymers else 'none requested; homopolymer screen not performed'
    caption = 'Raw candidate scoring; ASO excluded motifs: ' + exclusions + '. Later filtering is recorded separately.'
    plt.gcf().text(0.5, -0.02, textwrap.fill(caption, width=110), ha='center', va='top', fontsize=8)
    plt.savefig(outputu, bbox_inches='tight')
    plt.close()

    print("Finished at:", datetime.now().strftime("%H:%M:%S"))
if __name__ == "__main__":
    sys.exit(main())
