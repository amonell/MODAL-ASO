# Example requests

Specify the target, species, desired effect and any constraints. MODAL-ASO asks for missing details when needed.

**Gapmer design**

```bash
modal-aso "Design 10 RNase H-active gapmer ASOs targeting human PCSK9 mRNA."
```

**Splice modulation**

```bash
modal-aso "Design steric splice-switching ASOs to increase exon 7 inclusion in human SMN2."
```

**Regulatory blocking**

```bash
modal-aso "Design steric ASOs to block translation initiation at the annotated start codon of human BCL2 mRNA."
```

**Evaluate supplied candidates**

```bash
modal-aso "Evaluate the ASOs in candidates.csv against human PCSK9. Include their target matches, chemistry and off-target evidence."
```

**Compare existing results**

```bash
modal-aso "Compare the candidate panels in run_a and run_b. Explain the main differences using their saved evidence."
```

Use paths to your own files or run directories. For an exact transcript, add its accession and version to the request.
