#!/usr/bin/env python
import numpy as np
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
import pyBigWig
import pysam
import statistics
import pandas as pd
from tqdm import trange
from Bio import SeqIO, bgzf
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction as GC
from datetime import datetime
from pytz import timezone, utc
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
pd.options.mode.chained_assignment = None

__all__ = []
__version__ = '1.0.0'
__date__ = '2024-02-13'
__updated__ = '2024-02-13'

DEBUG = 0
TESTRUN = 0
PROFILE = 0

utc_time = datetime.utcnow()
pst_tz = timezone('US/Pacific')
pst_time = utc_time.replace(tzinfo=utc).astimezone(pst_tz)
today = pst_time.strftime("%Y%m%d")

# copied from https://github.com/YeoLab/rbp-maps/blob/master/maps/density/ReadDensity.py
class Density:
    def values(self, chrom, start, end, strand):
        return 0
        
class Conservation(Density):

    def __init__(self, phastcon, name=None):
        try:
            self.phastcon = pyBigWig.open(phastcon)
            self.name = name if name is not None else os.path.basename(phastcon)
        except Exception as e:
            print("couldn't open the bigwig files!")
            print(e)

    def values(self, chrom, start, end, strand):
        """

        Parameters
        ----------
        chrom : basestring
            (eg. chr1)
        start : int
            0-based start (first position in chromosome is 0)
        end : int
            1-based end (last position is not included)
        strand : str
            either '+' or '-'

        Returns
        -------
        densites : list
            values corresponding to density over specified positions.
        """

        def _fetch(c):
            if strand == "+":
                return self.phastcon.values(c, start, end)
            elif strand == "-":
                return list(reversed(self.phastcon.values(c, start, end)))
            else:
                return 1

        try:
            return _fetch(chrom)
        except RuntimeError:
            # retry with/without chr prefix
            try:
                if not chrom.startswith('chr'):
                    return _fetch(f"chr{chrom}")
                else:
                    return _fetch(chrom[3:])
            except RuntimeError:
                return [np.NaN] * abs(start - end)

    def avg_value(self, chrom, start, end, strand):
        return self.summary(chrom, start, end, strand)[0]

    def summary(self, chrom, start, end, strand):
        values = [float(x) for x in self.values(chrom, start, end, strand)
                  if x is not None and np.isfinite(x)]
        return (sum(values) / len(values) if values else float('nan'),
                len(values), max(0, int(end) - int(start)))
    

class NeutralConservation(Density):
    """Legacy class name for unavailable, rather than imputed, annotations."""

    def __init__(self):
        self.value = float('nan')
        self.name = "unavailable_conservation"

    def values(self, chrom, start, end, strand):
        return [self.value] * max(0, abs(int(end) - int(start)))

    def avg_value(self, chrom, start, end, strand):
        return self.value

    def summary(self, chrom, start, end, strand):
        return self.value, 0, max(0, int(end) - int(start))


def get_sequence(chrom, start, end, strand, genome_fasta):
    def _fetch_pysam(c):
        fasta = pysam.FastaFile(genome_fasta)
        try:
            if c not in fasta.references:
                return None
            return fasta.fetch(c, int(start), int(end))
        except Exception:
            return None
        finally:
            fasta.close()

    def _fetch_bedtools(c):
        bed = pybedtools.BedTool(f"{c}\t{start}\t{end}\t.\t0\t{strand}", from_string=True)
        seq = bed.sequence(fi=genome_fasta, s=False)
        with open(seq.seqfn) as f:
            try:
                next(f)
                return next(f).strip()
            except StopIteration:
                return None

    def _try_fetch(c):
        seq = _fetch_pysam(c)
        if seq:
            return seq
        try:
            return _fetch_bedtools(c)
        except Exception:
            return None

    # Try as-is
    seq = _try_fetch(chrom)
    # Try adding/removing 'chr' prefix to reconcile UCSC vs Ensembl naming
    if not seq and not chrom.startswith('chr'):
        seq = _try_fetch(f"chr{chrom}")
    if not seq and chrom.startswith('chr'):
        seq = _try_fetch(chrom[3:])
    if not seq:
        raise RuntimeError(f"Chromosome {chrom} not found in FASTA; tried with and without 'chr' prefix.")
    return seq


def tile(tile_length, bedtool, genome_fasta, output, pc, ph, compressed=True):
    tiled_seqs = []
    
    for region in bedtool:
        max_value = 0
        new_position = ""
        chrom = region.chrom
        start = region.start
        end = region.end
        name = region.name
        strand = region.strand
    
        seq = get_sequence(chrom, start, end, strand, genome_fasta)        
        
        progress = trange((len(seq) - tile_length) + 1, desc='parsing tiles', position=0, leave=False)
        for i in range((len(seq) - tile_length) + 1):
            if strand == '+':
                new_seq = Seq(seq[i:i + tile_length])
            elif strand == '-':
                new_seq = Seq(seq[i:i + tile_length]).reverse_complement()
                
            new_position = f"{chrom}:{start + i + 1}-{start + i + tile_length}:{strand}"  # position reported is 1-based
            gc_content = GC(new_seq)
            avg_conservation, pc_covered, _ = pc.summary(chrom, start + i, start + i + tile_length, strand)
            avg_phylop, ph_covered, _ = ph.summary(chrom, start + i, start + i + tile_length, strand)
            new_id = (f"id={name}|pos={new_position}|gc={gc_content:.2f}|pc={avg_conservation:.2f}|ph={avg_phylop:.2f}"
                      f"|pc_covered_bases={pc_covered}|ph_covered_bases={ph_covered}|conservation_interval_bases={tile_length}")
            
            tiled_seqs.append(SeqRecord(Seq(new_seq), id=f"{new_id}", description=""))
            progress.update(1)

    if compressed:
        with bgzf.BgzfWriter(output, 'wb') as output:
            SeqIO.write(tiled_seqs, output, "fasta")
    else:
        with open(output, 'w') as output:
            SeqIO.write(tiled_seqs, output, "fasta")
    

def main(argv=None):  # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (
        program_version, program_build_date
    )
    program_shortdesc = (__import__("__main__").__doc__ or "Tile sequences from BED regions").split("\n")[0]
    program_license = '''%s

    Created by Brian Yee on %s.
    Copyright 2016 organization_name. All rights reserved.

    Licensed under the Apache License 2.0
    http://www.apache.org/licenses/LICENSE-2.0

    Distributed on an "AS IS" basis without warranties
    or conditions of any kind, either express or implied.
    ''' 
    #% (program_shortdesc, str(__date__))

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
        "-b", "--bed",
        dest="bed_file",
        help="bedfile containing regions to tile",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-l", "--length", "--len",
        dest="tile_length",
        help="length of tile (default: 20)",
        required=False,
        type=int,
        default=20,
    )
    parser.add_argument(
        "-f", "--fasta",
        dest="genome_fasta",
        help="Genome assembly sequence fasta format",
        required=True,
    )
    parser.add_argument(
        "-p", "--phastcons",
        dest="phastcons",
        help="phastcons file",
        required=False,
    )
    parser.add_argument(
        "-y", "--phylop",
        dest="phylop",
        help="phylop file",
        required=False,
    )
    parser.add_argument(
        "--no-conservation",
        dest="no_conservation",
        help="Use neutral conservation scores when species conservation tracks are unavailable.",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        help="Directory to save tiles to (default: current directory)",
        required=False,
        default=".",
    )
    args = parser.parse_args()
    bed_file = args.bed_file
    tile_length = args.tile_length
    genome_fasta = args.genome_fasta
    phastcons = args.phastcons
    phylop = args.phylop
    output_dir = args.output_dir
    
    print("Begin at:", datetime.now().strftime("%H:%M:%S"))

    if args.no_conservation:
        print("Conservation tracks disabled; annotations are unavailable.")
        pc = NeutralConservation()
        ph = NeutralConservation()
    else:
        if not phastcons or not phylop:
            raise ValueError("--phastcons and --phylop are required unless --no-conservation is set.")
        pc = Conservation(phastcons)
        ph = Conservation(phylop)
    
    bedtool = pybedtools.BedTool(bed_file)
    
    output = os.path.join(output_dir, os.path.splitext(os.path.basename(bed_file))[0] + ".fasta.gz")
    print(f"Tiling {os.path.basename(bed_file)} at:", datetime.now().strftime("%H:%M:%S"))
    tile(tile_length, bedtool, genome_fasta, output, pc, ph, compressed=True)
    print(f"Finished {os.path.basename(bed_file)} at:", datetime.now().strftime("%H:%M:%S"))
if __name__ == "__main__":
    sys.exit(main())
