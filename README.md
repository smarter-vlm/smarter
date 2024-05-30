To run training and eval, from the repo root (need at least 40GB mem, 16 cores on arch x86_64, and a GPU for triton dependency)

```bash
git clone https://github.com/D-Roberts/smarter.git
cd smarter
conda create --name smarter python=3.10
conda activate smarter
pip install -r requirements.txt

python main_reasoner.py --model_name fused_dinov2_siglip --log --word_embed siglip --save_root final_runs --data_root data/smart-data/SMART101-release-v1/SMART101-Data --lr 0.0003 --wd 0.2 --batch_size 128 --num_heads 2 --repr_size 128 --qf_layer --eps 1e-8 --beta2 0.98 --pdrop 0.2 --ln_eps 1e-6 --h_sz 256 --seed 0 --num_workers 16 
```


Experiments tracked in CometML

https://www.comet.com/droberts308/multimodalai/view/new/panels
https://www.comet.com/droberts308/vlm-reasoners/view/C6sw7GhOEifcK1S0eJL5i4rgx/panels

To be able to create other CometML plots, a Comet API Key must be created and placed in the <root_dir>/.comet_token file and a Comet account username must be written to  <root_dir>/.comet_workspace.

To download the SMART101 dataset (from [merl](https://github.com/merlresearch/SMART)), please execute the get_SMART_data.sh script in the repository. 


