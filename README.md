# mycv
A personal computer vision repository using pytorch

## Installation
0. Clone repository
1. `conda develop --name pytorch_env path/to/mycv`

## Uninstall
`conda develop --uninstall path/to/mycv`

## Imagenet experiments
Notes:
- Markdown table generator: https://www.tablesgenerator.com/markdown_tables
- Imagenet 'Corrupt EXIF data’: https://discuss.pytorch.org/t/corrupt-exif-data-messages-when-training-imagenet/17313

Resnet-50, imagenet200_600
| Run     | lr    | Optimizer    | Reduction | Plain best/last | EMA best/last |
|---------|-------|--------------|-----------|-----------------|---------------|
| DEFAULT | 0.001 | SGD          | sum       |                 |               |
| res50_0 | 0.01  | SGD          |           | 72.4/72.4       | 74.7/74.1     |
| res50_1 | 0.001 | SGD          |           | 75.3/75.3       | 78.1/76.5     |
| res50_5 |       | SGD+nesterov |           | 75.8/75.4       | 77.6/76.8     |
|         | 0.01  | SGD          | mean      | 72.9/72.9       | 75.7/74.8     |


Resnet-50, full imagenet
| Run     | lr    | lr schedule | Reduction | Batchsize | Epochs | Plain best/last | EMA best/last |
|---------|-------|-------------|-----------|-----------|--------|-----------------|---------------|
| DEFAULT | 0.1   | step 0.1,30 | mean      | 256       | 90     |                 |               |
| res50_1 | 0.001 | cosine 0.2  | sum       |           | 60/100 | 68.19/68.13     | 70.83/70.55   |
| res50_1 |       |             |           |           |        | 75.51/75.36     | 75.87/75.87   |
| res50   |       |             |           | 128x2 dp  |        | r               | r             |
| res50   |       |             |           | 128x2 ac  |        |                 |               |
