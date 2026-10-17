#!/usr/bin/env bash
set -euo pipefail

# Download reference assets for ASO design.
# Assets vary by species (default: human / GRCh38). Use --species mouse to fetch GRCm39/mm39 (phastCons35way/phyloP35way).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
BOWTIE_DIR="${DATA_DIR}/bowtie"
TMP_DIR="${DATA_DIR}/tmp"

usage() {
	cat <<'EOF'
Usage: download_data.sh [--species <human|mouse>] [--data-dir <path>] [--help]

Downloads reference assets required by the ASO tiling pipeline.
Species defaults to human (GRCh38/hg38); pass --species mouse for GRCm39/mm39.
By default files are written to ./data; --data-dir writes the same expected layout elsewhere.
Requires Bowtie 1 (`bowtie` and `bowtie-build`); this script does not download Bowtie 2 indexes.
EOF
}

SPECIES_INPUT="human"
DATA_DIR_INPUT=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		-s|--species)
			if [[ $# -lt 2 ]]; then
				echo "Error: --species requires a value" >&2
				usage
				exit 1
			fi
			SPECIES_INPUT="$2"
			shift 2
			;;
		--data-dir)
			if [[ $# -lt 2 ]]; then
				echo "Error: --data-dir requires a value" >&2
				usage
				exit 1
			fi
			DATA_DIR_INPUT="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Error: unknown argument '$1'" >&2
			usage
			exit 1
			;;
	esac
done

normalize_species() {
	local value
	value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
	case "$value" in
		human|hg38|grch38) echo "human" ;;
		mouse|mm39|grcm39) echo "mouse" ;;
		*)
			echo ""
			;;
	esac
}

SPECIES_KEY="$(normalize_species "${SPECIES_INPUT}")"
if [[ -z "${SPECIES_KEY}" ]]; then
	echo "Error: unsupported species '${SPECIES_INPUT}'. Supported: human, mouse." >&2
	exit 1
fi

if [[ -n "${DATA_DIR_INPUT}" ]]; then
	DATA_DIR="${DATA_DIR_INPUT}"
	if [[ "${DATA_DIR}" != /* ]]; then
		DATA_DIR="${PWD}/${DATA_DIR}"
	fi
	BOWTIE_DIR="${DATA_DIR}/bowtie"
	TMP_DIR="${DATA_DIR}/tmp"
fi

if [[ "${SPECIES_KEY}" == "human" ]]; then
	UCSC_BUILD="hg38"
	GENOME_BUILD="GRCh38"
	PHASTCONS_TRACK="phastCons30way"
	PHYLOP_TRACK="phyloP30way"
	CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.chrom.sizes"
	PHASTCONS_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons30way/hg38.phastCons30way.bw"
	PHYLOP_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP30way/hg38.phyloP30way.bw"
	GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_42/gencode.v42.annotation.gtf.gz"
	GTF_GZ_OUT="${DATA_DIR}/gencode.v42.annotation.gtf.gz"
	GTF_OUT="${DATA_DIR}/gencode.v42.annotation.gtf"
	REFSEQ_XREF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_42/gencode.v42.metadata.RefSeq.gz"
	REFSEQ_XREF_OUT="${DATA_DIR}/gencode.v42.metadata.RefSeq.gz"
	GENOME_GZ_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"
	GENOME_GZ_OUT="${DATA_DIR}/GRCh38_no_alt_analysis_set_GCA_000001405.15.fna.gz"
	GENOME_FASTA_OUT="${DATA_DIR}/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
	BT1_ZIP_URL="ftp://ftp.ccb.jhu.edu/pub/data/bowtie_indexes/GRCh38_no_alt.zip"
	BT1_ZIP_OUT="${BOWTIE_DIR}/GRCh38_no_alt.bt1.zip"
	BT1_DIR="${BOWTIE_DIR}/GRCh38_no_alt_bt1"
elif [[ "${SPECIES_KEY}" == "mouse" ]]; then
	UCSC_BUILD="mm39"
	GENOME_BUILD="GRCm39"
	PHASTCONS_TRACK="phastCons35way"
	PHYLOP_TRACK="phyloP35way"
	CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.chrom.sizes"
	PHASTCONS_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phastCons35way/mm39.phastCons35way.bw"
	PHYLOP_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phyloP35way/mm39.phyloP35way.bw"
	GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M35/gencode.vM35.primary_assembly.annotation.gtf.gz"
	GTF_GZ_OUT="${DATA_DIR}/gencode.vM35.primary_assembly.annotation.gtf.gz"
	GTF_OUT="${DATA_DIR}/gencode.vM35.primary_assembly.annotation.gtf"
	REFSEQ_XREF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M35/gencode.vM35.metadata.RefSeq.gz"
	REFSEQ_XREF_OUT="${DATA_DIR}/gencode.vM35.metadata.RefSeq.gz"
	GENOME_GZ_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M35/GRCm39.primary_assembly.genome.fa.gz"
	GENOME_GZ_OUT="${DATA_DIR}/GRCm39.primary_assembly.genome.fa.gz"
	GENOME_FASTA_OUT="${DATA_DIR}/GRCm39.primary_assembly.genome.fa"
	BT1_ZIP_URL=""
	BT1_ZIP_OUT=""
	BT1_DIR=""
else
	echo "Error: unexpected species key '${SPECIES_KEY}'" >&2
	exit 1
fi

PHASTCONS_OUT="${DATA_DIR}/${UCSC_BUILD}.${PHASTCONS_TRACK}.bw"
PHYLOP_OUT="${DATA_DIR}/${UCSC_BUILD}.${PHYLOP_TRACK}.bw"
CHROM_SIZES_OUT="${DATA_DIR}/${UCSC_BUILD}.chrom.sizes"
BOWTIE_ALIAS_NAME="${UCSC_BUILD}"

mkdir -p "${DATA_DIR}" "${BOWTIE_DIR}" "${TMP_DIR}"

need_cmd() {
	command -v "$1" >/dev/null 2>&1 || {
		echo "Error: required command '$1' not found in PATH" >&2
		exit 1
	}
}

# Use curl if available, else wget
_download() {
	local url="$1"
	local out="$2"
	if [ -s "$out" ]; then
		echo "Already exists: $out"
		return 0
	fi
	if command -v curl >/dev/null 2>&1; then
		if curl -L -C - --fail --retry 3 --retry-delay 5 -o "$out" "$url"; then
			return 0
		fi
		echo "Error: download failed for $url" >&2
		return 1
	elif command -v wget >/dev/null 2>&1; then
		if wget -c -O "$out" "$url"; then
			return 0
		fi
		echo "Error: download failed for $url" >&2
		return 1
	else
		echo "Error: need either curl or wget installed" >&2
		exit 1
	fi
	echo "Error: download failed for $url" >&2
	return 1
}

need_cmd gzip

REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

refresh_bowtie_path() {
	if command -v brew >/dev/null 2>&1; then
		local prefix
		prefix="$(brew --prefix bowtie 2>/dev/null || true)"
		if [[ -n "${prefix}" && -d "${prefix}/bin" ]]; then
			export PATH="${prefix}/bin:${PATH}"
		fi
	fi
}

ensure_bowtie_1() {
	refresh_bowtie_path
	if command -v bowtie >/dev/null 2>&1 && command -v bowtie-build >/dev/null 2>&1; then
		return 0
	fi
	local py="${PYTHON:-}"
	if [[ -z "${py}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
		py="${REPO_ROOT}/.venv/bin/python"
	fi
	if [[ -z "${py}" ]] && command -v python3 >/dev/null 2>&1; then
		py="python3"
	fi
	if [[ -n "${py}" ]]; then
		echo "Ensuring Bowtie 1 is installed before downloading reference assets..."
		(cd "${REPO_ROOT}" && "${py}" -m aso_agents.runtime_binaries) || true
		refresh_bowtie_path
		if command -v bowtie >/dev/null 2>&1 && command -v bowtie-build >/dev/null 2>&1; then
			return 0
		fi
	fi
    if command -v brew >/dev/null 2>&1; then
		echo "Installing Bowtie 1 with Homebrew (brewsci/bio/bowtie)..."
		brew install brewsci/bio/bowtie
		refresh_bowtie_path
	fi
	if command -v bowtie >/dev/null 2>&1 && command -v bowtie-build >/dev/null 2>&1; then
		return 0
	fi
	cat >&2 <<EOF
Error: Bowtie 1 is required (bowtie and bowtie-build) before reference assets can be finished.
Install it first, then re-run this script:
  macOS:          brew install brewsci/bio/bowtie
  Debian/Ubuntu:  sudo apt-get install -y bowtie
  conda:          conda install -y -c bioconda bowtie
EOF
	exit 1
}

ensure_bowtie_1

# 1) Chrom sizes (UCSC)
if [[ -n "${CHROM_SIZES_URL}" ]]; then
	_download "$CHROM_SIZES_URL" "$CHROM_SIZES_OUT"
fi

# 2) PhastCons bigWig (UCSC)
if [[ -n "${PHASTCONS_URL}" ]]; then
	_download "$PHASTCONS_URL" "$PHASTCONS_OUT"
fi

# 2b) phyloP bigWig (UCSC)
if [[ -n "${PHYLOP_URL}" ]]; then
	_download "$PHYLOP_URL" "$PHYLOP_OUT"
fi

# 2c) GENCODE GTF
if [[ -n "${GTF_URL}" ]]; then
	_download "$GTF_URL" "$GTF_GZ_OUT"
	if [ ! -s "$GTF_OUT" ]; then
		echo "Decompressing GTF..."
		gzip -cd "$GTF_GZ_OUT" > "$GTF_OUT"
	fi
fi

# 2d) Version-paired GENCODE RefSeq transcript cross-references
if [[ -n "${REFSEQ_XREF_URL}" ]]; then
	_download "$REFSEQ_XREF_URL" "$REFSEQ_XREF_OUT"
fi

# 3) Primary genome FASTA
if [[ -n "${GENOME_GZ_URL}" ]]; then
	_download "$GENOME_GZ_URL" "$GENOME_GZ_OUT"
	if [ ! -s "$GENOME_FASTA_OUT" ]; then
		echo "Decompressing genome FASTA..."
		gzip -cd "$GENOME_GZ_OUT" > "$GENOME_FASTA_OUT"
	fi
fi

# 4) Bowtie 1 indexes (.ebwt). This pipeline does not use Bowtie 2.
if [[ -n "${BT1_ZIP_URL}" ]]; then
	if command -v unzip >/dev/null 2>&1; then
		if [ ! -d "$BT1_DIR" ]; then
			if _download "$BT1_ZIP_URL" "$BT1_ZIP_OUT"; then
				if unzip -t "$BT1_ZIP_OUT" >/dev/null 2>&1; then
					mkdir -p "$BT1_DIR"
					unzip -oq "$BT1_ZIP_OUT" -d "$BT1_DIR"
				else
					echo "Ignoring incomplete or invalid Bowtie 1 zip; will build the index locally."
					rm -f "$BT1_ZIP_OUT"
				fi
			else
				echo "Prebuilt Bowtie 1 index download failed; will build the index locally with bowtie-build."
				rm -f "$BT1_ZIP_OUT"
			fi
		fi
	else
		echo "Note: 'unzip' not found; skipping Bowtie 1 prebuilt index and building locally."
	fi
fi

# Create species alias symlinks based on whichever Bowtie 1 index exists
create_symlinks() {
	local prefix="$1"; shift
	local kind="$1"; shift  # "bt1"
	local alias="$1"; shift
	local dest_dir="$BOWTIE_DIR"
	if [ "$kind" = "bt1" ]; then
		for suf in 1 2 3 4 rev.1 rev.2; do
			src="${prefix}.${suf}.ebwt"
			dst="${dest_dir}/${alias}.${suf}.ebwt"
			if [ -f "$src" ] && [ ! -L "$dst" ]; then
				ln -sf "$src" "$dst"
			fi
		done
	fi
}

# Prefer bt1 if found
bt1_prefix_file=""
if [ -d "$BT1_DIR" ]; then
	bt1_prefix_file="$(find "$BT1_DIR" -maxdepth 2 -type f -name "*.1.ebwt" | head -n1 || true)"
fi
if [ -n "$bt1_prefix_file" ]; then
	create_symlinks "${bt1_prefix_file%.1.ebwt}" "bt1" "${BOWTIE_ALIAS_NAME}"
	BOWTIE_BASE_MSG="${BOWTIE_DIR}/${BOWTIE_ALIAS_NAME} (Bowtie 1)"
else
	echo "No prebuilt Bowtie 1 indexes were linked; building locally."
	BOWTIE_BASE_MSG="<none>"
fi

# Ensure Bowtie index exists or attempt local build
if [[ "${BOWTIE_BASE_MSG}" == "<none>" ]]; then
	if command -v bowtie-build >/dev/null 2>&1; then
		if [ ! -s "${GENOME_FASTA_OUT}" ]; then
			echo "Error: Bowtie index missing and genome FASTA not found at ${GENOME_FASTA_OUT}." >&2
			exit 1
		fi
		echo "Building Bowtie 1 index locally for ${GENOME_BUILD} (this may take several minutes)..."
		mkdir -p "${BOWTIE_DIR}"
		if bowtie-build "${GENOME_FASTA_OUT}" "${BOWTIE_DIR}/${BOWTIE_ALIAS_NAME}"; then
			BOWTIE_BASE_MSG="${BOWTIE_DIR}/${BOWTIE_ALIAS_NAME} (Bowtie 1 - local build)"
		else
			echo "Error: bowtie-build failed; Bowtie index remains unavailable." >&2
			exit 1
		fi
	else
		cat >&2 <<EOF
Error: Bowtie 1 index not available for ${SPECIES_KEY}, and 'bowtie-build' is not installed.
Please install Bowtie (https://bowtie-bio.sourceforge.net/index.shtml) and run:
  bowtie-build ${GENOME_FASTA_OUT} ${BOWTIE_DIR}/${BOWTIE_ALIAS_NAME}
Then re-run this script.
EOF
		exit 1
	fi
fi

echo ""
echo "All downloads complete for species '${SPECIES_KEY}' (${GENOME_BUILD}/${UCSC_BUILD}). Files are in: ${DATA_DIR}"
echo "Bowtie index basename is: ${BOWTIE_BASE_MSG}"
