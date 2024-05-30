
## Code and requirements

```bash
git clone https://github.com/D-Roberts/smarter.git
cd smarter
conda create --name smarter python=3.10
conda activate smarter
pip install -r requirements.txt
```

## SMART101 Data
To be able to run a small experiment without the need to download the full dataset, two puzzles are committed to the repo. See the Small Runs section of the README to see how to run a toy train and eval experiment tracked in CometML.

To download the full SMART101 dataset (from [merl](https://github.com/merlresearch/SMART)), please execute the get_SMART_data.sh script in the repository. Depending on the internet connection, it can take 1-5hrs to download.
 


## Small Runs
To run training and eval of smaller models on a subset of the dataset which is committed to this repo for initial insights into the deep learning training of vision-language reasoners, from the repo root (need an arch x86_64):

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
--puzzles 16,58

rm -rf small-data
rm -rf small-runs
```


The dataset is sampled from two puzzles, 16,58, illustrated in the article, one from the math skill with multiple choice answer and one from the path skill with sequence answer.

Args are described in main_reasoner.py.

## Machine learning experiment tracking with CometML

Experiments are tracked in CometML. A public account is made available for trying out the code, and experiment panels (loss and accuracy curves) can be seen here [https://www.comet.com/ai-daor/smarter/view/new/panels](https://www.comet.com/ai-daor/smarter/view/new/panels).

To be able to create personal experiments, a Comet API Key must be created and placed in the <root_dir>/.comet_token file and a Comet account username must be written to  <root_dir>/.comet_workspace, replacing the public one (from [CometML](https://www.comet.com)).


## Final Models
To run training and eval of final models, from the repo root (need at least 40GB mem, 16 cores on arch x86_64, and a V100 GPU (or A100 or H100)):


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




