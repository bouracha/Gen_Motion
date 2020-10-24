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
