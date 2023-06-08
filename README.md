# The Limitations of Multivariate Transformer based models in Time Series Forecasting

## Install
Follow these instructions to clone the repo and prepare the datasets:

1. Clone this repo:
    ```
    git clone https://github.com/anon285/univariate-vs-multivariate.git
    ```

2. From within the project directory, run the `prepare_datasets.sh` bash script:
    ```
    cd univariate-vs-multivariate
    bash prepare_datasets.sh
    ```
3. Install these requirements via pip for `Python 3.9`:
    ```
    einops==0.6.0
    matplotlib==3.6.2
    numpy==1.23.4
    pandas==1.5.2
    pytorch_lightning==1.8.5
    requests==2.28.1
    scikit_learn==1.2.1
    scipy==1.10.1
    seaborn==0.12.2
    sympy==1.11.1
    torch==1.13.0
    tqdm==4.64.1
    ```

## Run experiments
Bash scripts for each experiment can be found in `code/experiments`.  Any experiment must be run from the `code` directory:
```
cd code
bash experiments/univariate/vt_all_datasets.sh
```
This example will train a Vanilla Transformer model for each dataset in our univariate setting.  Results will be printed in the terminal.  Additionally, all results are saved in `code/results` in a Pickled Pandas DataFrame for batch processing by the user.
