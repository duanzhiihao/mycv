# mycv
A personal computer vision repository using pytorch

## Installation
0. Clone repository
1. `conda develop --name pytorch_env path/to/mycv`

## Uninstall
`conda develop --uninstall path/to/mycv`

Notes: https://www.tablesgenerator.com/markdown_tables

## Imagenet experiments
Resnet-50, imagenet200_600
| Run     | lr    | Optimizer    | Plain best/last | EMA best/last |
|---------|-------|--------------|-----------------|---------------|
| default | 0.001 | SGD          |                 |               |
| res50_0 | 0.01  | SGD          | 72.4/72.4       | 74.7/74.1     |
| res50_1 | 0.001 | SGD          | 75.3/75.3       | 78.1/76.5     |
|         |       | SGD+nesterov | 75.8/75.4       | 77.6/76.8     |
