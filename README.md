 [On Vision-Language Reasoners Technical Report](https://drive.google.com/file/d/1cyJM1csY-CME_lVycioj0GDX4CtFsTgY/view?usp=sharing)


To run training and eval (need at least 40GB mem):

```
conda create --name smarter python=3.10
conda activate smarter
pip install -r requirements.txt

python main_reasoner.py --model_name fused_dinov2_siglip --log --word_embed siglip --save_root /home/hice1/droberts308/scratch/final_runs --data_root <SMART101_data_path> --lr 0.0003 --wd 0.2 --batch_size 128 --num_heads 2 --repr_size 128 --qf_layer --eps 1e-8 --beta2 0.98 --pdrop 0.2 --ln_eps 1e-6 --h_sz 256 --seed 0 --num_workers 16 
```


Experiments tracked in CometML

https://www.comet.com/droberts308/multimodalai/view/new/panels
https://www.comet.com/droberts308/vlm-reasoners/view/C6sw7GhOEifcK1S0eJL5i4rgx/panels

To be able to create your own CometML plots, you must place your Comet API Key in the modules/denisa_vlm_reasoners/.comet_token file and your Comet account user in  modules/denisa_vlm_reasoners/.comet_workspace.

To download the SMART101 dataset (from [merl](https://github.com/merlresearch/SMART)), please execute the get_SMART_data.sh script in the repository folder "scripts". 


