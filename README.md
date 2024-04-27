 This is the step-by-step code I am working with to architect and train the vlm reasoners step by step for the final project DL 7643. PRs and Comet runs for all changes.

Experiment tracked in CometML

 https://www.comet.com/droberts308/multimodalai/view/new/panels
 https://www.comet.com/droberts308/vlm-reasoners/view/C6sw7GhOEifcK1S0eJL5i4rgx/panels

To run training and eval (need at least 40GB mem):

```
conda create --name smarter python=3.10
conda activate smarter
pip install -r requirements.txt

python main_reasoner.py --model_name fused_dinov2_siglip --log --word_embed siglip --save_root /home/hice1/droberts308/scratch/final_runs --data_root /home/hice1/droberts308/scratch/smart/SMART101-release-v1/SMART101-Data/ --lr 0.0003 --wd 0.2 --batch_size 128 --num_heads 2 --repr_size 128 --qf_layer --eps 1e-8 --beta2 0.98 --pdrop 0.2 --ln_eps 1e-6 --h_sz 256 --seed 0 --num_workers 16 
```