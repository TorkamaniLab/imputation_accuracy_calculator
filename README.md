# A tool for calculating imputation accuracy by comparing imputation results to WGS data.

Documentation and scripts provided for calculating imputation accuracy. Tested using Minimac4 and Imputation Autoencoder (vcf format imputed files).

## Requirements

- samtools/bcftools: used to calculate MAFs
- python v3: source code was implemented and tested on python 3.6

## Required command line arguments are:

The following inputs, in vcf.gz format, including its respective tabix tbi file, are required to run.

- ga: genotype array file used as input for imputation
- imputed: imputation results
- wgs: ground truth file, containing experimentally determined genotypes (i.e. Whole Genome Sequencing data)

All files provided must be in vcf.gz format (compressed, tabixed). Alleles must match, in other words: NO SWAPS, NO FLIPS, SAME BUILD!!! It is not necessary to provide allele frequencies, since the tool will calculate it internally using bcftools.

## Usage:

The help shows all the required arguments listed above, plus optional arguments.

```
python3 Compare_imputation_to_WGS.py -h
usage: Compare_imputation_to_WGS.py --ga <input_genotype_array.vcf.gz> --imputed <imputed_file.vcf.gz> --wgs <whole_genome_file.vcf.gz>
Use -h or --help to display help.

arguments:
  -h, --help            show this help message and exit
  --ga GA               path to genotype array file in vcf.gz format, with tbi
  --wgs WGS             path to whole genome file in vcf.gz format, with tbi
  --imputed IMPUTED     path to imputed file in vcf.gz format, with tbi
  --ref REF             optional, path to reference panel file in vcf.gz
                        format, with tbi. Used for MAF calculation. WGS file
                        will be used if no reference file is provided.
  --max_total_rows MAX_TOTAL_ROWS
                        maximun number of rows or variants to be loaded
                        simultaneously, summing all chunks loaded by all cores
  --max_per_core MAX_PER_CORE
                        maximun number of variants per chunk per core, lower
                        it to avoid RAM overload
  --min_per_core MIN_PER_CORE
                        minimun number of variants per chunk per core,
                        increase to avoid interprocess communication overload
  --sout SOUT           optional output file path/name per sample, default is
                        the same as the imputed file with
                        _per_sample_results.txt suffix
  --vout VOUT           optional output file path/name per variant, default is
                        the same as the imputed file with
                        _per_variant_results.txt suffix
```

A detailed report with accuracy ratio, F1 score, Pearson correlation (r2) is generated and wrote to the output file (i.e accuracy_result.txt)

## How to run example:
```
python3.6 Compare_imputation_to_WGS.py --ga aric_GT_ancestry-5_cad_190625.vcf.chr1.gz --imputed aric_intersectWGS_rpt-1_ancestry-5_phasing-eagle_seed-E_imputed-HRC_cad_190625.vcf.chr1.gz --wgs aric_WGS_ancestry-5_cad_190625.recode.vcf.chr1.gz
```

## Results:

```
Processing  239 imputed samples
Processing chunk: 1 Max rows per chunk: 10000
1 Read imputed file time:  0.0004201233386993408
2 Chunking time:  8.239876478910446e-06
3 Calculation time:  0.16841873712837696
4 Merging calculations per sample time:  0.0037479093298316
Results per sample at: aric_intersectWGS_rpt-1_ancestry-5_phasing-eagle_seed-E_imputed-HRC_cad_190625.vcf.chr1_per_sample_results.txt
Results per variant at: aric_intersectWGS_rpt-1_ancestry-5_phasing-eagle_seed-E_imputed-HRC_cad_190625.vcf.chr1_per_variant_results.txt
Total run time (sec): 0.17389075085520744

```

The results will be displayed as the example bellow (per variant):
```
position        SNP     REF_MAF IMPUTED_MAF     WGS_MAF F-score concordance_P0  IQS     r2      precision       recall  TP      TN      FP      FN
22:48742786     22:48742786_T_C 0.0045  0.0     0.0053  0.995   0.989   0.0     0.011   1.0     0.989   1.0     0.989   0.0     0.011
22:48741824     22:48741824_A_G 0.0002  0.0     0.002   0.998   0.996   0.0     0.001   1.0     0.996   1.0     0.996   0.0     0.004
22:48742525     22:48742525_G_A 0.0041  0.0013  0.0033  0.998   0.996   0.57    0.36    1.0     0.996   1.0     0.996   0.0     0.004
22:48742178     22:48742178_G_A 0.0089  0.0007  0.0153  0.986   0.971   0.081   0.062   1.0     0.972   1.0     0.971   0.0     0.029
22:48742575     22:48742575_G_A 0.0001  0.0     0.0     1.0     1.0     0.0     1.0     1.0     1.0     1.0     1.0     0.0     0.0
22:48742286     22:48742286_G_A 0.0001  0.0007  0.0007  0.999   0.997   -0.001  0.0     0.999   0.999   0.999   0.999   0.001   0.001
22:48742498     22:48742498_G_A 0.0028  0.0     0.002   0.998   0.996   0.0     0.014   1.0     0.996   1.0     0.996   0.0     0.004
22:48741615     22:48741615_C_T 0.0002  0.0     0.0007  0.999   0.999   0.0     0.0     1.0     0.999   1.0     0.999   0.0     0.001
22:48741402     22:48741402_A_G 0.2096  0.0439  0.2646  0.836   0.621   0.233   0.185   0.879   0.796   0.879   0.64    0.121   0.36
```

Results per sample:
```
imputed_ids     WGS_ids F-score concordance_P0  r2      precision       recall  TP      TN      FP      FN
A00003_A00003   A00003_A00003   0.976   0.95    0.497   0.972   0.981   0.972   0.979   0.028   0.021
A00018_A00018   A00018_A00018   0.932   0.851   0.696   0.972   0.896   0.972   0.875   0.028   0.125
A00056_A00056   A00056_A00056   0.99    0.98    0.613   0.981   1.0     0.981   1.0     0.019   0.0
A00080_A00080   A00080_A00080   0.986   0.97    0.671   0.981   0.991   0.981   0.989   0.019   0.011
A00083_A00083   A00083_A00083   0.99    0.98    0.56    0.981   1.0     0.981   1.0     0.019   0.0
A00099_A00099   A00099_A00099   0.99    0.98    0.139   0.99    0.99    0.99    0.99    0.01    0.01
A00120_A00120   A00120_A00120   0.976   0.96    0.5     0.962   0.99    0.962   0.99    0.038   0.01
A00146_A00146   A00146_A00146   0.972   0.941   0.64    0.963   0.981   0.963   0.979   0.037   0.021
A00152_A00152   A00152_A00152   0.986   0.97    0.403   0.981   0.99    0.981   0.99    0.019   0.01
```

The results can be interpreted as follows.

Metrics per variant:
- REF_MAF: Reference Panel MAF (if reference panel is provided)
- IMPUTED_MAF: Imputed MAF
- WGS_MAF: Whole Genome MAF
- F-score: macro F-score (not weighted),
- concordance_P0: accuracy ratio (concordance, from the article cited bellow [1]),
- IQS: imputation quality score (from the same article [1])
- precision: precision
- recall: recall
- TP: true positives
- TN: true negatives
- FP: false positives
- FN: false negatives

Metrics per sample:
- F-core: F-score per sample
- concordance_P0: accuracy ratio
- r2: r-squared
- precision: precision
- recall: recall
- TP: true positives
- TN: true negatives
- FP: false positives
- FN: false negatives

## References:

[1] Ramnarine S, Zhang J, Chen LS, Culverhouse R, Duan W, Hancock DB, Hartz SM, Johnson EO, Olfson E, Schwantes-An TH, Saccone NL. When does choice of accuracy measure alter imputation accuracy assessments?. PloS one. 2015 Oct 12;10(10):e0137601.
