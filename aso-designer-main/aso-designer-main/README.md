# aso-designer
Use various filters to design ASOs.

# Filters:
- GC%
- Sequences (eg. CCCCC, GGGGG)
- Mapping
- Secondary structure
- Conservation (filter by phastcon)

# Workflow:
- For each BED file, for each region in BED file, generate overlapping tiles ```{TILE_LEN}```
- Map each tiles with bowtie (```bowtie -a -f -v {params.num_mismatches} -y -m 3 -l 18 --sam --sam-nohead --no-unal --seed 1 {params.bowtie_index} {input.tiles} {output.mapped}```)
- Converts SAM -> FASTA format (and aggregate multi-mapping tiles)
- Run RNAfold to determine secondary structure (```RNAfold --noconv --noPS {input.aligned} > {output.folded}```)
- Score tiles according to pre-defined filters
- Upload trackhub to UCSC (need access)

# Usage (Yeo lab only): 

## Inputs:

You will need at least one BED6 file (the example runs two BED files). These BED files should describe regions where tiles are made from:
```
chr2	96536742	96552634	ENSG00000196843.17	0	+
```

You will also need a config file that specifies the locations of the BED file(s) and paths to reference data. Filter params can also be set here. You can also re-run the workflow with updated parameters to re-score: 


#### **Testing**: `CHEMISTRY` supports adding modifications to each tile (See: [Outputs](https://github.com/byee4/aso-designer?tab=readme-ov-file#outputs) for IDT-style output)
Options:

- **5-10-5**: A 5-10-5 antisense oligonucleotide (ASO) is a type of gapmer ASO with a central stretch of 10 DNA nucleotides flanked by five modified ribonucleotides on each end. This design is commonly used to downregulate gene expression. 
- **steric-hindrance**: 2' MOE modification applied to all nucleotides. 
- **3-10-3**: Affinity Plus bases are typically used in shorter ASOs in a 3-10-3 gapmer format (three Affinity Plus bases, ten DNA bases, three Affinity Plus bases). The shorter length of Affinity Plus ASOs is beneficial when delivering the ASO in vivo without a delivery tool.


Config file:
```
# input folder where regions files below are found
INPUT_FOLDER: /tscc/projects/ps-yeolab4/software/aso/2.0.0/examples 
# list of regions to tile/query, in BED6 format
REGIONS:  
  - EXAMPLE1.bed
  - EXAMPLE2.bed
  
# define output folder
OUTPUT_FOLDER: outputs  

# reference data
GENOME: /tscc/projects/ps-yeolab4/software/aso/2.0.0/examples/reference/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
PHASTCONS: /tscc/projects/ps-yeolab4/software/aso/2.0.0/examples/reference/hg38.phastCons30way.bw
BOWTIE_INDEX: /tscc/projects/ps-yeolab4/software/aso/2.0.0/examples/reference/bowtie/hg38
CHROM_SIZES: /tscc/projects/ps-yeolab4/software/aso/2.0.0/examples/reference/hg38.chrom.sizes
UCSC_SPECIES: hg38

# filter parameters
TILE_LEN: 20
CHEMISTRY: '5-10-5' # can also be 'steric-hindrance', or '3-10-3'

# float value between 0 and 1 (1 = most conserved)
CONSERVATION: 0.99
# use 2 if you want to be more stringent with mapping. This will consider any alignment with up to two mismatches (and remove them if the alignment is not unique).
NUM_MISMATCHES: 1  
# GC% filters, keep tiles between GC_MIN and GC_MAX
GC_MIN: 0.4
GC_MAX: 0.7
# filter tiles containing any sequence in this list
AVOID: 
  - CCCCC
  - GGGGG
```

### Reference data sources

- Genome & chrom sizes: 
  - [GRCh38.fasta](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/)
- Bowtie index:
  - [Pre-built indices](https://bowtie-bio.sourceforge.net/manual.shtml)
- Phastcons files (use the .bw bigwig files):
  - [phastcons7way human+vertebrates](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons7way/)
  - [phastcons30way humans+primates](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons30way/)
  - [phastcons100way humans+vertebrates](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/)
  
## Example run command:
```
# For heavy/high throughput workloads, use an interactive node.
module load aso/3.0.0;
mkdir -p /tscc/lustre/ddn/scratch/${USER}/aso-test/v3.0.0;
cd /tscc/lustre/ddn/scratch/${USER}/aso-test;
cp $ASO_HOME/examples/run_example.sh ./
cp $ASO_HOME/examples/config.yaml ./
sh run_example.sh
```

# Outputs:
- Outputs from two dummy BED file samples can be found in this repository, under "```examples/outputs```". The primary output will be called "SAMPLE_NAME.scored.bed, whose format is described below:
- SAMPLE_NAME.scored.bed - full list of tiles in BED6 format.
  - The "**name**" column encodes the actual information used to apply each filter.
  - The "**score**" column encodes how many filters were passed.
  - A tile can have a maximum score of 1000.
  - Each filter that did not pass reduces a tile's score by 200 (minimum: 0).
  - Tiles that map to multiple hits will have their score reduced by 100 for each additional hit. For example, a tile that passes all other filters but maps to two loci will have a score of 900.
- Upon pipeline completion, top scoring (score=1000) tiles will be printed along with a link to a UCSC trackhub:

```
### TOP HITS FOR outputs/ARID5A_2.scored.bed ###
chr4    145134535       145134555       gc=0.45|pc=0.97|mappedto=chr4:145134536-145134556|nostruct=YES  1000    -       CACAAAGGAAACAACACGCT
chr4    145134536       145134556       gc=0.45|pc=0.97|mappedto=chr4:145134537-145134557|nostruct=YES  1000    -       ACAAAGGAAACAACACGCTG
chr4    145134536       145134556       gc=0.45|pc=0.97|mappedto=chr12:3845382-3845402|nostruct=YES     1000    -       CAGCGTGTTGTTTCCTTTGT
chr4    145134537       145134557       gc=0.50|pc=0.97|mappedto=chr12:3845381-3845401|nostruct=YES     1000    -       GCAGCGTGTTGTTTCCTTTG
URL: https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&hubUrl=http://s3-us-west-2.amazonaws.com/yeolab-trackhubs/20250128-asodesign-EXAMPLE2/20250128-asodesign-EXAMPLE2.hub.txt&ignoreCookie=1&wgEncodeGencodeV42_hideKids=1&wgEncodeGencodeCompV42=full&wgEncodeGencodeCompV42_sel=1&position=chr4:145133685-145134648
```
- **Testing** New output: EXAMPLE1.scored.idt.txt - "IDT" style table with chemical modifications. This table is tab-separated and contains three columns:
  - IDT: Synthesis sequence with chemical modification attached
  - Synthesis: The reverse-complement sequence of the tile
  - Sequence: The original tile sequence
  
## UCSC Trackhub:
To view tiles on a genome browser, use the provided URL from the finished workflow. You will need to "show" the track for it to appear:
![alt text](img/screenshot1.png)

You can click on the track itself to modify the display mode to show each tile, as well as filter out tiles that fail a score threshold:
![alt text](img/screenshot2.png)

![alt text](img/screenshot3.png)

Zooming into specific regions allow reveal specific tiles and filter metadata:
![alt text](img/screenshot4.png)

# Frequently asked questions

**Q:** Can you put multiple transcripts in at once? Like i put 4 transcripts in one bed? Or will it not tell me which transcript it is targeting?

**A:** If you want separate transcripts (one track per transcript), you would have four BED files. But if you wanted to say, have one transcript with four distinct exons, you would have four regions in one bed.

**Q:** What does the `NUM_MISMATCHES` parameter control for, exactly?

**A:** Say we have one tile "AAAA" that aligns to a single position chr1:1-4 uniquely (no other AAAA in genome). Setting `NUM_MISMATCHES` to 0 would report a single uniquely mapped hit. 

However, say there are two other positions:
1. chr1:101-104 with sequence AAAG and
2. chr1:201-204 with sequence AAGG.

- Setting `NUM_MISMATCHES` to 1 would report two non-uniquely mapped hits (chr1:1-4, chr1:101-104), or one off-target.
- Setting `NUM_MISMATCHES` to 2 would report three non-uniquely mapped hits or two off-targets (chr1:1-4, chr1:101-104, chr1:201-204).
- Only by setting `NUM_MISMATCHES` to 0 would report a successfully passed filter (no off-targets) for sequence AAAA.
