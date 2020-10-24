## Generative Model-Enhanced Human Motion Prediction
This is the code for the paper



Anthony Bourached, Ryan-Rhys Griffiths, Robert Gray, Ashwani Jha, Parashkev Nachev.
[_Generative Model-Enhanced Human Motion Prediction_](https://arxiv.org/abs/2010.11699). Under review at ICLR 2021.


### Dependencies

* cuda 9.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.3.1.
* [progress 1.5](https://pypi.org/project/progress/)

### Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

### Training commands
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train on angle space, in-distribution, H3.6M:
```bash
python main.py --data_dir "[Path To Your H36M data]/h3.6m/dataset/" --variational True --lambda 0.003 --n_z 8 --dropout 0.3 --lr_gamma 1.0 --input_n 10 --output_n 10 --dct_n 20
```
in-distribution (CMU):
```bash
python main.py --dataset 'cmu_mocap' --data_dir "[Path To Your CMU data]/cmu_mocap/" --variational True --lambda 0.003 --n_z 8 --dropout 0.3 --lr_gamma 1.0 --input_n 10 --output_n 25 --dct_n 35
```
to train on 3D space for CMU, simply change the ```--dataset 'cmu_mocap'``` to ```--dataset 'cmu_mocap_3d```. This flag is 'h3.6m' by default.

To train on 'walking' and test out-of-distribution (for h3.6M), include the extra flag:
```bash
--out_of_distribution 'walking' 
```
identically to train on 'basketball' and test out-of-distribution (for CMU), include the extra flag:
```bash
--out_of_distribution 'basketball' 
```
The same models may be trained (or used for inference independent of how they were trained) without the VGAE branch by changing the 
```
--variational True
``` 
flag to 
```
--variational False
```
(True by default).

### Citing

If you use our code, and/or build on our work, please cite our paper:

```
@misc{bourached2020generative,
      title={Generative Model-Enhanced Human Motion Prediction}, 
      author={Anthony Bourached and Ryan-Rhys Griffiths and Robert Gray and Ashwani Jha and Parashkev Nachev},
      year={2020},
      eprint={2010.11699},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgments

The codebase is built on that of https://github.com/wei-mao-2019/LearnTrajDep and depends heavily on their work in [_Learning Trajectory Dependencies for Human Motion Prediction_](https://arxiv.org/abs/1908.05436) (ICCV 2019), and [_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://arxiv.org/abs/2007.11755) (ECCV 2020). Thus please also cite:

```
@inproceedings{wei2019motion,
  title={Learning Trajectory Dependencies for Human Motion Prediction},
  author={Wei, Mao and Miaomiao, Liu and Mathieu, Salzemann and Hongdong, Li},
  booktitle={ICCV},
  year={2019}
}
```

and

```
@article{mao2020history,
  title={History Repeats Itself: Human Motion Prediction via Motion Attention},
  author={Mao, Wei and Liu, Miaomiao and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2007.11755},
  year={2020}
}
```

### Licence
MIT
