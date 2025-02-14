
## Code and requirements

Using Conda:

```bash
git clone https://github.com/D-Roberts/smarter.git
cd smarter
conda create --name smarter python=3.10
conda activate smarter
pip install -r requirements.txt
```

Using `venv`

```bash
python3 -m venv smarter-env
source ./smarter-env/bin/activate
pip install -r requirements.txt
```

To install conda if necessary, can do [miniconda](https://docs.anaconda.com/free/miniconda/).


## Small Runs
To be able to run a small experiment without the need to download the full dataset, a math puzzle is committed to the repo. See the *SMART101 Data* section of the README to see how to download the full dataset to run full final models train and eval.

To run training and eval of smaller models on the small subset of the dataset which is committed to this repo for initial insights into the deep learning training of vision-language reasoners, from the repo root:

```bash
unzip small-data.zip

python main_reasoner.py \
--model_name fused_dinov2_siglip \
--log \
--word_embed siglip \
--save_root small-runs \
--data_root small-data/SMART101-Data \
--lr 0.0001 \
--wd 0.2 \
--batch_size 4 \
--num_heads 2 \
--repr_size 128 \
--qf_layer \
--eps 1e-8 \
--beta2 0.98 \
--pdrop 0.2 \
--ln_eps 1e-6 \
--h_sz 64 \
--seed 0 \
--num_workers 1 \
--data_tot 128 \
--train_diff easy \
--num_epochs 2 \
--puzzles 58

rm -rf small-data
rm -rf small-runs
```


The small dataset has one puzzle, 58, illustrated in the article, from the math skill with multiple choice answer. Note that the skill-based accuracies will not be calculated since at least 3 puzzles are necessary.

Args are described in *main_reasoner.py*.

## Machine learning experiment tracking with CometML

Experiments are tracked in CometML. A public account is made available for trying out the code, and experiment panels (loss and accuracy curves) can be seen here [https://www.comet.com/ai-daor/smarter/view/new/panels](https://www.comet.com/ai-daor/smarter/view/new/panels).

To be able to create personal experiments, a Comet API Key must be created and placed in the smarter/.comet_token file and a Comet account username must be written to smarter/.comet_workspace (from your [CometML](https://www.comet.com) account), replacing the public one.

## SMART101 Data
To download the full SMART101 dataset (from [merl](https://github.com/merlresearch/SMART)), please execute the `get_SMART_data.sh` script in the repository. Depending on the internet connection, it can take 1-5hrs to download.


## Final Models
To run training and eval of final models, from the repo root (need at least 40GB mem, at least 16 cores, and a V100 GPU (or A100 or H100)):


```bash
python main_reasoner.py \
--model_name fused_dinov2_siglip \
--log \
--word_embed siglip \
--save_root final-runs \
--data_root data/smart-data/SMART101-release-v1/SMART101-Data \
--lr 0.0003 \
--wd 0.2 \
--batch_size 128 \
--num_heads 2 \
--repr_size 128 \
--qf_layer \
--eps 1e-8 \
--beta2 0.98 \
--pdrop 0.2 \
--ln_eps 1e-6 \
--h_sz 256 \
--seed 0 \
--num_workers 16 \
--data_tot 1000 \
--train_diff easy \
--num_epochs 3 \
--puzzles all
```

Tested on Ubuntu20.04 LTS, Mac (x86_64 and M1) CPU-only, V100, A100, H100.


