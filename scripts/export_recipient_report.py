"""Export a new versioned presentation with explicit chemistry; never redesign.

The caller specifies field bindings and chemistry/status. A contradictory
original synthesis string causes failure rather than re-encoding a molecule.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aso_recipient_report import attest_record, write_recipient_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','id-column','sequence-column','notation-column','architecture','assumption-status','title'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--rank-column');p.add_argument('--linkages', required=True, help='Comma-separated PS/PO, or explicitly full-PS.')
    p.add_argument('--context', required=True)
    a = p.parse_args();source = Path(a.input); records=[]
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            sequence = row[a.sequence_column]
            links = ['PS']*(len(sequence)-1) if a.linkages == 'full-PS' else a.linkages.split(',')
            rank = int(row[a.rank_column]) if a.rank_column else None
            records.append(attest_record(row[a.id_column], sequence, a.architecture, row[a.notation_column], linkages=links, assumption_status=a.assumption_status, native_rank=rank))
    write_recipient_report(a.output,records,title=a.title,context=a.context,provenance={'source_filename':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'presentation':'Clearly labeled format copy; no molecular or rank changes.'})

if __name__ == '__main__':main()
