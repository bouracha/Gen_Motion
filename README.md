# Generative Motion


## Dependencies
Some older versions may work. But we used the following:

* cuda 10.1
* Python 3.6.9
* [Pytorch](https://github.com/pytorch/pytorch) 1.6.0
* [progress 1.5](https://pypi.org/project/progress/)

## Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

## Training commands
To train on 3D H3.6M for different latent space sizes:
```bash
python3 main.py
```
To train as VAE add:
```
--variational
``` 
flag.


## Licence

MIT
