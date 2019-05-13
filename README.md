# A tool for calculating imputation accuracy by comparing imputation results to WGS data.

Documentation and scripts provided for calculating imputation accuracy. Tested using Minimac4 and Imputation Autoencoder (vcf format imputed files).

## Command line arguments are:

- ref_file: reference panel used for imputation
- input file: genotype array file used as input for imputation
- imputed_file: imputation results
- WGS_file: ground truth file, containing experimentally determined genotypes (i.e. Whole Genome Sequencing data)
- accuracy_result.txt: name for the output file, can be any name by user's choice

All files provided must be in vcf format, uncompressed, positions and alleles must match.

## How to run:
```
python3.6 Compare_imputation_to_WGS.py ref_file input_file imputed_file WGS_file > accuracy_result.txt
```

A detailed report with accuracy ratio, F1 score, Pearson correlation (r2) is generated and wrote to the output file (i.e accuracy_result.txt)

## Example for ARIC dataset using Minimac4, 9p21.3 region only:
```
python3.6 Compare_imputation_to_WGS.py \
HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4 \
ARIC_PLINK_flagged_chromosomal_abnormalities_zeroed_out_bed.lifted_NCBI36_to_GRCh37.GH.ancestry-1.chr9_intersect1.vcf.gz.9p21.3.recode.vcf \
imputed_9_intersect1.dose.9p21.3.vcf \
c1_ARIC_WGS_Freeze3.lifted_already_GRCh37_intersect1.vcf.gz.9p21.3.recode.vcf \
> accuracy_ARIC_minimac4.txt
```

## Results:

The accuracy metrics are split by MAF thresholds: full dataset, MAF<0.005, MAF>0.005. They can be extracted easily:
```
tail -n 13 accuracy_ARIC_minimac4.txt
```

The results will be displayed as the example bellow: 
```
LABEL: acc_0-1 acc_0-0.005 acc_0.005-1
ACC_RESULT: 0.9564710679945054 0.9978051518374098 0.8466405023547883
ACC_RESULT_STDERR: 0.003999562406265492 0.0010794785390533794 0.012592857010616916
LABEL: F1_0-1 F1_0-0.005 F1_0.005-1
F1_RESULT: 0.961353586638197 0.998362734174821 0.8772014118603704
F1_RESULT_STDERR: 0.0 0.0 0.0
F1_RESULT_MICRO: 0.8062435549490555 0.8157781916861591 0.7809086630476098
F1_RESULT_MICRO_STDERR: 0.36839909745187904 0.3867015575778903 0.3132287255888532
F1_RESULT_MACRO: 0.9590986936591761 0.9967534702651305 0.8735426739758072
F1_RESULT_MACRO_STDERR: 0.0007125480730197961 0.001073935051745486 0.002014179072566615
LABEL: r2_0-1 r2_0-0.005 r2_0.005-1
R2_RESULT: 0.9206789637146933 0.9972854432692814 0.699405645114335
R2_RESULT_STDERR: 0.0015275731841606893 2.4734013322507845e-05 0.005542140635649253
```

The results can be interpreted as follows:

- ACC_RESULT: accuracy ratio, number of correct predictions divided by total number of predictions
- ACC_RESULT_STDERR: standard error of accuracy ratio acrross all variants
- F1_RESULT: overall F1 score across all genotypes
- F1_RESULT_STDERR: standard error cannot be calculated for overall F1 score
- F1_RESULT_MICRO: micro F1 score, averaged accross all samples
- F1_RESULT_MICRO_STDERR: standard error of the micro F1 score
- F1_RESULT_MACRO: macro F1 score, averaged accross all features
- F1_RESULT_MACRO_STDERR:  standard error of the macro F1 score
- R2_RESULT: mean Pearson r squared per sample
- R2_RESULT_STDERR: standard error of the Pearson r squared


You can extract a specific metrix from the resul file using grep:
```
grep "R2_RESULT:" accuracy_*.txt > r2_result.txt
grep "R2_RESULT_" accuracy_*.txt > r2_stderr_result.txt
```

