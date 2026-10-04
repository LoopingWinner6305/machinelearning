# Dataset source and attribution

`Decision_trees/processed.cleveland.data` is the processed Cleveland file supplied in the original learning repository. It has 303 rows and 14 columns including the target. Its local SHA-256 is `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`. The file bytes were preserved during this cleanup; the upstream download has not been independently byte-compared.

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).
Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

UCI lists this dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This data licence is separate from the code's licensing status. Source and licence checked on 4 October 2026.

The experiment converts target values 1 through 4 to 1 and retains 0 as 0. The file is not modified; missing markers are parsed and imputed inside training pipelines. Other notebooks use synthetic toy data generated in code. These historical records do not establish suitability for clinical diagnosis or current populations.
