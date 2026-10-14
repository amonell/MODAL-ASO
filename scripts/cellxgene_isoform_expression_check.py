#!/usr/bin/env python3
"""Query gene expression from CellXGene Census without downloading full datasets.

The Census API allows filtering by organism, tissue, cell type, and gene,
returning only the data you need.

Examples:
    # List available tissues for human
    python cellxgene_isoform_expression_check.py --list-tissues --species human

    # List available cell types in brain
    python cellxgene_isoform_expression_check.py --list-cell-types --species human --tissue brain

    # Query expression for specific genes in brain tissue
    python cellxgene_isoform_expression_check.py --species human --tissue brain --gene MAPT --gene APP --gene PSEN1

    # Query with cell type filter and sampling
    python cellxgene_isoform_expression_check.py --species human --tissue brain --cell-type neuron --gene MAPT --max-cells 50000

    # Query by Ensembl ID
    python cellxgene_isoform_expression_check.py --species human --gene ENSG00000186868

    # Output as JSON
    python cellxgene_isoform_expression_check.py --species human --tissue brain --gene MAPT --format json

    # AxisQuery test (first 1000 cells in subset)
    python cellxgene_isoform_expression_check.py --axis-query --species human --tissue brain --cell-type neuron --gene ENSG00000186868 --max-cells 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def open_census(version: str = "stable"):
    """Open CellXGene Census connection."""
    try:
        import cellxgene_census
    except ImportError as e:
        print("Error: cellxgene-census not installed. Run `uv sync` from the repository root.", file=sys.stderr)
        sys.exit(1)
    return cellxgene_census.open_soma(census_version=version)


def list_organisms(census) -> List[str]:
    """List available organisms in the census."""
    return list(census["census_data"].keys())


def normalize_organism(species: str) -> str:
    """Normalize species name to census organism key."""
    mapping = {
        "human": "homo_sapiens",
        "homo sapiens": "homo_sapiens",
        "mouse": "mus_musculus",
        "mus musculus": "mus_musculus",
    }
    key = species.lower().strip()
    return mapping.get(key, key)


def list_tissues(census, organism: str) -> List[Tuple[str, int]]:
    """List available tissue_general values with cell counts."""
    obs = census["census_data"][organism].obs
    df = obs.read(column_names=["tissue_general"]).concat().to_pandas()
    counts = df["tissue_general"].value_counts()
    return [(tissue, int(count)) for tissue, count in counts.items()]


def list_cell_types(census, organism: str, tissue: Optional[str] = None) -> List[Tuple[str, int]]:
    """List available cell_type values with counts, optionally filtered by tissue."""
    obs = census["census_data"][organism].obs
    columns = ["cell_type"]
    value_filter = None
    if tissue:
        value_filter = f"tissue_general == '{tissue}'"
    df = obs.read(column_names=columns, value_filter=value_filter).concat().to_pandas()
    counts = df["cell_type"].value_counts()
    return [(ct, int(count)) for ct, count in counts.head(100).items()]


def list_datasets(census, organism: str, tissue: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available datasets with counts."""
    obs = census["census_data"][organism].obs
    columns = ["dataset_id"]
    value_filter = None
    if tissue:
        value_filter = f"tissue_general == '{tissue}'"
    df = obs.read(column_names=columns, value_filter=value_filter).concat().to_pandas()
    counts = df["dataset_id"].value_counts()
    return [{"dataset_id": did, "cell_count": int(count)} for did, count in counts.head(50).items()]


def find_genes(census, organism: str, gene_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Find genes by name or Ensembl ID, returning their soma_joinid and metadata."""
    var = census["census_data"][organism].ms["RNA"].var
    results = {}

    for gene in gene_names:
        gene_clean = gene.strip()
        if not gene_clean:
            continue

        # Try feature_name first, then feature_id
        if gene_clean.upper().startswith("ENS"):
            # Ensembl ID - try with and without version
            base_id = gene_clean.split(".")[0]
            filter_expr = f"feature_id == '{gene_clean}'"
            df = var.read(value_filter=filter_expr).concat().to_pandas()
            if df.empty and base_id != gene_clean:
                filter_expr = f"feature_id == '{base_id}'"
                df = var.read(value_filter=filter_expr).concat().to_pandas()
        else:
            filter_expr = f"feature_name == '{gene_clean}'"
            df = var.read(value_filter=filter_expr).concat().to_pandas()

        if not df.empty:
            row = df.iloc[0]
            results[gene_clean] = {
                "soma_joinid": int(row["soma_joinid"]),
                "feature_id": row["feature_id"],
                "feature_name": row["feature_name"],
                "feature_length": int(row.get("feature_length", 0)) if "feature_length" in row else None,
            }

    return results


def query_expression(
    census,
    organism: str,
    gene_names: List[str],
    tissue: Optional[str] = None,
    cell_type: Optional[str] = None,
    dataset_id: Optional[str] = None,
    max_cells: Optional[int] = None,
    sample_seed: Optional[int] = 42,
) -> Dict[str, Dict[str, Any]]:
    """Query mean expression for genes with optional filtering.

    Returns dict mapping gene_name -> {mean_expression, fraction_expressing, n_cells, ...}
    """
    try:
        import cellxgene_census
    except ImportError:
        raise RuntimeError("cellxgene-census required")

    # Build observation filter
    filters = []
    if tissue:
        filters.append(f"tissue_general == '{tissue}'")
    if cell_type:
        # Use 'in' for partial match on cell type
        filters.append(f"cell_type == '{cell_type}'")
    if dataset_id:
        filters.append(f"dataset_id == '{dataset_id}'")

    obs_filter = " and ".join(filters) if filters else None

    # Build gene filter
    gene_filters = []
    for gene in gene_names:
        gene_clean = gene.strip()
        if not gene_clean:
            continue
        if gene_clean.upper().startswith("ENS"):
            base_id = gene_clean.split(".")[0]
            gene_filters.append(f"feature_id == '{gene_clean}'")
            if base_id != gene_clean:
                gene_filters.append(f"feature_id == '{base_id}'")
        else:
            gene_filters.append(f"feature_name == '{gene_clean}'")

    var_filter = " or ".join(gene_filters) if gene_filters else None

    if not var_filter:
        return {}

    # Sample cells if needed
    obs_coords = None
    total_cells = None
    sampled_cells = None

    if max_cells:
        # First count total cells matching filter
        obs_df = cellxgene_census.get_obs(
            census,
            organism=organism,
            value_filter=obs_filter,
            column_names=["soma_joinid"],
        )
        total_cells = len(obs_df)

        if total_cells > max_cells:
            rng = np.random.default_rng(sample_seed)
            joinids = obs_df["soma_joinid"].to_numpy()
            sample = rng.choice(joinids, size=max_cells, replace=False)
            obs_coords = list(sample)
            sampled_cells = max_cells

    # Get expression data
    adata = cellxgene_census.get_anndata(
        census,
        organism=organism,
        obs_value_filter=obs_filter,
        obs_coords=obs_coords,
        var_value_filter=var_filter,
        X_name="raw",
    )

    n_cells = adata.n_obs
    if total_cells is None:
        total_cells = n_cells

    # Calculate expression stats per gene
    results = {}
    feature_names = list(adata.var["feature_name"])
    feature_ids = list(adata.var["feature_id"])

    for i, (fname, fid) in enumerate(zip(feature_names, feature_ids)):
        col = adata.X[:, i]
        if hasattr(col, "toarray"):
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()

        mean_expr = float(np.mean(col))
        frac_expr = float(np.sum(col > 0) / len(col)) if len(col) > 0 else 0.0
        sum_expr = float(np.sum(col))

        # Map back to input gene name
        matched_name = fname
        for gene in gene_names:
            if gene.upper() == fname.upper() or gene.upper() == fid.upper() or gene.upper().split(".")[0] == fid.upper().split(".")[0]:
                matched_name = gene
                break

        results[matched_name] = {
            "feature_name": fname,
            "feature_id": fid,
            "mean_expression": mean_expr,
            "sum_expression": sum_expr,
            "fraction_expressing": frac_expr,
            "n_cells": n_cells,
            "total_cells": total_cells,
            "sampled_cells": sampled_cells,
            "tissue": tissue,
            "cell_type": cell_type,
            "dataset_id": dataset_id,
        }

    return results


def query_expression_axis_query(
    census,
    organism: str,
    gene_names: List[str],
    tissue: Optional[str] = None,
    cell_type: Optional[str] = None,
    dataset_id: Optional[str] = None,
    max_cells: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Query mean expression using SOMA axis_query (no AnnData materialization)."""
    try:
        import tiledbsoma as soma
    except ImportError:
        raise RuntimeError("tiledbsoma is required for axis_query")

    exp = census["census_data"][organism]

    filters = []
    if tissue:
        filters.append(f"tissue_general == '{tissue}'")
    if cell_type:
        filters.append(f"cell_type == '{cell_type}'")
    if dataset_id:
        filters.append(f"dataset_id == '{dataset_id}'")
    obs_filter = " and ".join(filters) if filters else None

    gene_filters = []
    for gene in gene_names:
        gene_clean = gene.strip()
        if not gene_clean:
            continue
        if gene_clean.upper().startswith("ENS"):
            base_id = gene_clean.split(".")[0]
            gene_filters.append(f"feature_id == '{gene_clean}'")
            if base_id != gene_clean:
                gene_filters.append(f"feature_id == '{base_id}'")
        else:
            gene_filters.append(f"feature_name == '{gene_clean}'")
    var_filter = " or ".join(gene_filters) if gene_filters else None
    if not var_filter:
        return {}

    obs_query = soma.AxisQuery(value_filter=obs_filter) if obs_filter else soma.AxisQuery()
    var_query = soma.AxisQuery(value_filter=var_filter)

    total_cells = None
    sampled_cells = None
    obs_coords = None

    if max_cells:
        obs_start = time.time()
        obs_df = exp.obs.read(
            value_filter=obs_filter,
            column_names=["soma_joinid"],
        ).concat().to_pandas()
        total_cells = len(obs_df)
        obs_elapsed = time.time() - obs_start
        print(f"[axis_query] obs read {total_cells} cells in {obs_elapsed:.1f}s")

        if total_cells > max_cells:
            if sample_seed is not None:
                rng = np.random.default_rng(sample_seed)
                obs_coords = rng.choice(obs_df["soma_joinid"].to_numpy(), size=max_cells, replace=False).tolist()
            else:
                obs_coords = obs_df["soma_joinid"].to_numpy()[:max_cells].tolist()
            sampled_cells = len(obs_coords)
        else:
            obs_coords = obs_df["soma_joinid"].to_numpy().tolist()
            sampled_cells = len(obs_coords)

    if obs_coords is not None:
        obs_query = soma.AxisQuery(coords=(obs_coords,))

    query = exp.axis_query(
        measurement_name="RNA",
        obs_query=obs_query,
        var_query=var_query,
    )

    if total_cells is None:
        try:
            total_cells = int(query.n_obs)
        except Exception:
            total_cells = len(query.obs_joinids().to_numpy())
    n_cells = sampled_cells if sampled_cells is not None else total_cells

    var_df = exp.ms["RNA"].var.read(
        value_filter=var_filter,
        column_names=["soma_joinid", "feature_id", "feature_name"],
    ).concat().to_pandas()
    if var_df.empty:
        return {}

    joinid_to_meta = {
        int(row["soma_joinid"]): {"feature_id": row["feature_id"], "feature_name": row["feature_name"]}
        for _, row in var_df.iterrows()
    }

    sums = {gid: 0.0 for gid in joinid_to_meta.keys()}
    counts = {gid: 0 for gid in joinid_to_meta.keys()}

    batch_start = time.time()
    chunk_count = 0
    for tbl in query.X("raw").tables():
        chunk_count += 1
        df = tbl.to_pandas()
        if df.empty:
            continue
        grouped = df.groupby("soma_dim_1")["soma_data"].agg(["sum", "count"])
        for gid, row in grouped.iterrows():
            gid_int = int(gid)
            sums[gid_int] += float(row["sum"])
            counts[gid_int] += int(row["count"])
    batch_elapsed = time.time() - batch_start
    print(f"[axis_query] X read {chunk_count} chunks in {batch_elapsed:.1f}s")

    results = {}
    for gid, meta in joinid_to_meta.items():
        expr_sum = sums.get(gid, 0.0)
        n_expressing = counts.get(gid, 0)
        mean_expr = expr_sum / n_cells if n_cells else 0.0
        frac_expr = n_expressing / n_cells if n_cells else 0.0

        matched_name = meta["feature_name"]
        for gene in gene_names:
            if gene.upper() == meta["feature_name"].upper() or gene.upper() == meta["feature_id"].upper():
                matched_name = gene
                break

        results[matched_name] = {
            "feature_name": meta["feature_name"],
            "feature_id": meta["feature_id"],
            "mean_expression": float(mean_expr),
            "sum_expression": float(expr_sum),
            "fraction_expressing": float(frac_expr),
            "n_cells": n_cells,
            "total_cells": total_cells,
            "sampled_cells": sampled_cells,
            "tissue": tissue,
            "cell_type": cell_type,
            "dataset_id": dataset_id,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query gene expression from CellXGene Census without downloading datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Discovery options
    parser.add_argument("--list-organisms", action="store_true", help="List available organisms")
    parser.add_argument("--list-tissues", action="store_true", help="List available tissues for species")
    parser.add_argument("--list-cell-types", action="store_true", help="List cell types (optionally filtered by tissue)")
    parser.add_argument("--list-datasets", action="store_true", help="List datasets (optionally filtered by tissue)")
    parser.add_argument("--find-gene", action="append", dest="find_genes", default=[], help="Find gene metadata by name/ID")

    # Query parameters
    parser.add_argument("--species", default="human", help="Species: human, mouse, or census organism key")
    parser.add_argument("--tissue", help="Filter by tissue_general (e.g., brain, heart, lung)")
    parser.add_argument("--cell-type", help="Filter by cell_type (e.g., neuron, astrocyte)")
    parser.add_argument("--dataset-id", help="Filter by specific dataset_id")
    parser.add_argument("--gene", action="append", dest="genes", default=[], help="Gene name or Ensembl ID to query (repeatable)")
    parser.add_argument("--gene-file", help="File with one gene per line")
    parser.add_argument("--axis-query", action="store_true", help="Use SOMA axis_query instead of AnnData")

    # Sampling
    parser.add_argument("--max-cells", type=int, default=100000, help="Max cells to sample (default: 100000)")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")

    # Output
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--census-version", default="stable", help="Census version (default: stable)")

    args = parser.parse_args()

    # Open census
    with open_census(args.census_version) as census:
        organism = normalize_organism(args.species)

        # Discovery commands
        if args.list_organisms:
            organisms = list_organisms(census)
            print("Available organisms:")
            for org in organisms:
                print(f"  {org}")
            return 0

        if args.list_tissues:
            tissues = list_tissues(census, organism)
            print(f"Tissues for {organism} (top by cell count):")
            for tissue, count in sorted(tissues, key=lambda x: -x[1])[:30]:
                print(f"  {tissue}: {count:,} cells")
            return 0

        if args.list_cell_types:
            cell_types = list_cell_types(census, organism, args.tissue)
            tissue_str = f" in {args.tissue}" if args.tissue else ""
            print(f"Cell types for {organism}{tissue_str} (top 100 by count):")
            for ct, count in cell_types:
                print(f"  {ct}: {count:,} cells")
            return 0

        if args.list_datasets:
            datasets = list_datasets(census, organism, args.tissue)
            tissue_str = f" in {args.tissue}" if args.tissue else ""
            print(f"Datasets for {organism}{tissue_str} (top 50 by count):")
            for ds in datasets:
                print(f"  {ds['dataset_id']}: {ds['cell_count']:,} cells")
            return 0

        if args.find_genes:
            gene_info = find_genes(census, organism, args.find_genes)
            if args.format == "json":
                print(json.dumps(gene_info, indent=2))
            else:
                print(f"Gene info for {organism}:")
                for gene, info in gene_info.items():
                    print(f"  {gene}:")
                    print(f"    feature_id: {info['feature_id']}")
                    print(f"    feature_name: {info['feature_name']}")
                    print(f"    soma_joinid: {info['soma_joinid']}")
            return 0

        # Expression query
        genes = list(args.genes)
        if args.gene_file:
            with open(args.gene_file) as f:
                genes.extend([line.strip() for line in f if line.strip()])

        if not genes:
            parser.print_help()
            print("\nError: No genes specified. Use --gene or --gene-file.", file=sys.stderr)
            return 1

        if args.axis_query:
            results = query_expression_axis_query(
                census,
                organism=organism,
                gene_names=genes,
                tissue=args.tissue,
                cell_type=args.cell_type,
                dataset_id=args.dataset_id,
                max_cells=args.max_cells,
                sample_seed=args.sample_seed,
            )
        else:
            results = query_expression(
                census,
                organism=organism,
                gene_names=genes,
                tissue=args.tissue,
                cell_type=args.cell_type,
                dataset_id=args.dataset_id,
                max_cells=args.max_cells,
                sample_seed=args.sample_seed,
            )

        # Output results
        output_handle = open(args.output, "w") if args.output else sys.stdout

        if args.format == "json":
            json.dump(results, output_handle, indent=2)
            output_handle.write("\n")
        elif args.format == "csv":
            import csv
            if results:
                fieldnames = ["gene", "feature_name", "feature_id", "mean_expression", "fraction_expressing", "n_cells", "total_cells", "tissue", "cell_type"]
                writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                writer.writeheader()
                for gene, info in results.items():
                    writer.writerow({
                        "gene": gene,
                        "feature_name": info["feature_name"],
                        "feature_id": info["feature_id"],
                        "mean_expression": f"{info['mean_expression']:.6g}",
                        "fraction_expressing": f"{info['fraction_expressing']:.4f}",
                        "n_cells": info["n_cells"],
                        "total_cells": info["total_cells"],
                        "tissue": info["tissue"] or "",
                        "cell_type": info["cell_type"] or "",
                    })
        else:
            # Table format
            tissue_str = f", tissue={args.tissue}" if args.tissue else ""
            cell_type_str = f", cell_type={args.cell_type}" if args.cell_type else ""

            if results:
                first_result = next(iter(results.values()))
                n_cells = first_result["n_cells"]
                total_cells = first_result["total_cells"]
                sampled = first_result.get("sampled_cells")
                sample_str = f" (sampled from {total_cells:,})" if sampled else ""
                print(f"Expression for {organism}{tissue_str}{cell_type_str}")
                print(f"Cells: {n_cells:,}{sample_str}")
                print()
                print(f"{'Gene':<20} {'Mean Expr':>12} {'Frac Expr':>12} {'Feature ID':<20}")
                print("-" * 70)
                for gene, info in results.items():
                    print(f"{gene:<20} {info['mean_expression']:>12.4f} {info['fraction_expressing']:>12.4f} {info['feature_id']:<20}")
            else:
                print("No results found for the specified genes.")

        if args.output:
            output_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
