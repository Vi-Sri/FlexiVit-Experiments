The experiments are for CAP5415 - Computer Vision, Probing FlexiVIT's patch representation power when finetuning, 
training from scratch and applying knowledge distillation to see if patch representation power is transferable

Required Libraries : 
flexivit-pytorch

```
git clone https://github.com/bwconrad/flexivit
cd flexivit/
pip install -r requirements.txt
```

These has all the plotting functions : 
plot_mad.py
plot_tsne.py
plot_attn.py 
visualize_linear_projections.py 

These are the training codes :
flexivit.py - Fine tune FlexiVit on CIFAR10
flexivit_kd.py - Train flexvit with Knowledge distillation 
flexivit_scratch.py - Train flexivit from scratch
