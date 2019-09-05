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
position        SNP     IMPUTED_MAF     WGS_MAF F-score concordance_P0  IQS     r2
1:38461319      1:38461319      0.2573  0.2615  0.896   0.715   0.478   0.38
1:56962821      1:56962821      0.159   0.1548  0.974   0.933   0.843   0.801
1:230845794     1:230845794     0.1381  0.1318  0.995   0.987   0.967   0.954
1:115753482     1:115753482     0.0397  0.0397  1.0     1.0     1.0     1.0
1:154422067     1:154422067     0.2782  0.2803  0.998   0.996   0.993   0.987
1:151762308     1:151762308     0.4895  0.4854  0.997   0.992   0.986   0.988
1:2252205       1:2252205       0.0084  0.0042  0.996   0.992   0.663   0.462
1:3325912       1:3325912       0.1381  0.1318  0.94    0.849   0.617   0.452
1:210468999     1:210468999     0.0084  0.0084  1.0     1.0     1.0     1.0
```

Results per sample:
```
imputed_ids     WGS_ids F-score concordance_P0  r2
0_326990        A00163_A00163   1.0     1.0     0.955
460_124582      A00250_A00250   0.957   0.889   0.856
0_120579        A00272_A00272   0.947   0.889   0.948
612_546283      A00412_A00412   0.957   0.889   0.709
0_116067        A00636_A00636   0.957   0.889   0.886
478_526103      A00796_A00796   1.0     1.0     0.971
690_433317      A00812_A00812   0.957   0.889   0.868
0_354504        A00948_A00948   0.9     0.778   0.905
0_530012        A00990_A00990   0.963   0.889   0.791
0_206281        A01096_A01096   1.0     1.0     0.995
612_277150      A01125_A01125   0.96    0.889   0.694
```

The results can be interpreted as follows.

Metrics per variant:
- REF_MAF: Reference Panel MAF (if reference panel is provided)
- IMPUTED_MAF: Imputed MAF
- WGS_MAF: Whole Genome MAF
- F-score: macro F-score (not weighted),
- concordance_P0: accuracy ratio (concordance, from the article cited bellow [1]),
- IQS: imputation quality score (from the same article [1])
- r2: r-squared

Metrics per sample:
- F-core: F-score per sample
- concordance_P0: accuracy ratio
- r2: r-squared

## References:

[1] Ramnarine S, Zhang J, Chen LS, Culverhouse R, Duan W, Hancock DB, Hartz SM, Johnson EO, Olfson E, Schwantes-An TH, Saccone NL. When does choice of accuracy measure alter imputation accuracy assessments?. PloS one. 2015 Oct 12;10(10):e0137601.
