# bmVAE

bmVAE is a variational autoencoder method for clustering single-cell mutation data.

## Requirements

* Python 3.7+.
* torch 1.11.0+.
* numpy.
* sklearn.

## Usage

bmVAE clusters cells into distinct subpopulations and infer genotypes of each subpopulation based on single-cell binary data.

Example:

```
python bmvae.py --input testdata/example.txt --output testdata
```

## Input Files

### Genotype Matrix

The SNVs of single cells are denoted as a genotype matrix. Each row defines the mutational states of a single cell, and each column represents a mutation. Columns are separated by tabs. The genotype matrix is binary.

The entry at position [i,j] should be

* 0 if mutation j is not observed in cell i,
* 1 if mutation j is observed in cell i, or
* 3 if the genotype information is missing

## Output Files

The output directory is provided by users.

### Genotypes of subpopulations

The genotypes of subpopulations are written to a file with name "clusters.txt".

### Cell labels

The cell-to-cluster assignments are written to a file with name "labels.txt".

### Estimated error rates

The estimated FPR and FNR are written to a file with name "para.txt".

## Arguments

* `--input <filename>` Replace \<filename\> with the file containing the genotype matrix.

* `--output <string>` Replace \<string\> with the output directory.

## Optional arguments

* `--Kmax <INT>` Set \<INT\> to a positive integer. This specifies the maximum number of clusters to consider. Default value is set to N/10 (N denotes the number of cells).

* `--seed <INT>` Set \<INT\> to a non-negative integer. This specifies the seed for generating random numbers. Default value is 0.

* `--epochs <INT>` Set \<INT\> to a positive integer. This specifies the number of epoches to train the VAE. Default value is 250.

* `--batch_size <INT>` Set \<INT\> to a positive integer. This specifies the batch size for training the VAE. Default value is 64.

* `--lr <Double>` Set \<Double\> to a positive real number. This specifies the learning rate for training the VAE. Default value is 0.0001.

* `--dimension <INT>` Set \<INT\> to a positive integer. This specifies the dimension of latent space of the VAE. Default value is 3.

## Contact

If you have any questions, please contact zhyu@nxu.edu.cn.