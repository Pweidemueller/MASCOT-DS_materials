#!/usr/bin/env python3

import argparse
from Bio import SeqIO

def process_alignment(input_file, output_file):
    """Replace N's with gaps in a FASTA alignment."""
    with open(output_file, "w") as out_f:
        for record in SeqIO.parse(input_file, "fasta"):
            record.seq = record.seq.upper().replace("N", "-").replace("n", "-")
            SeqIO.write(record, out_f, "fasta")
    print(f"Processed alignment saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Replace N characters with gaps in a FASTA alignment.')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA alignment file')
    parser.add_argument('-o', '--output', required=True, help='Output FASTA alignment file')
    
    args = parser.parse_args()
    process_alignment(args.input, args.output)

if __name__ == "__main__":
    main()
