# Timm experiments

Deep learning image classification experiments using the `timm` library.


## Experiment 1: Conditional batch normalization

During my master's thesis, I discovered that finetuning batch normalization layers of a ResNet model using a method called conditional batch normalization [1] was an effective method to finetune a handwriting recognition model which uses an encoder-decoder architecture (with the ResNet as the image backbone).

Standard batchnorm:

```
	y = weight * norm(x) + bias
```

Conditional batchnorm:

```
	y = (weight + wd) * norm(x) + (bias + bd)
```

where `wd` and `bd` are learned parameters that are added to the original batchnorm weights and biases. During training, only the `wd` and `bd` parameters are trained. In this case, I also train the final classification layer and freeze all other layers of the model frozen. The idea is that the batch normalization weight and bias parameters are the most effective parameters to finetune, while they make up only a small part of the the total number of parameters (see table below).

Here, I'm trying to find out: Can conditional batch normalization be successfully applied to finetune image recognition models on small datasets? More specifically: Can conditional batchnorm outperform end-to-end finetuning on very small image classification datasets?  

For reference:

| Model    | Total Parameters |  Conditional batchnorm parameters |
|----------|------------------|-----------------------------------|
| Resnet18 | 11,689,512       | 9,600 (+0.09%)                    |
| Resnet50 | 23,583,845       | 53,120 (+0.22%)                   |


### Dataset

Domain should probably be similar to pretraining: In my thesis, I started with a trained HTR model, and then adapted it to other writing styles.

I'm starting off with the [Oxford Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which is a well-balanced dataset with 37 classes, and as far as I can tell, pretty close to ImageNet. For a more comprehensive suite of dataset, I might try the VTAB Natural benchmark.

Useful links for exploring data:

- [Papers with code](https://paperswithcode.com/datasets?mod=images&q=ImageNet+dogs+vs+ImageNet+non-dogs&task=image-classification&page=1)
- [Tensorflow Know Your Data](https://knowyourdata-tfds.withgoogle.com/)

Alternative dataset candidates: 

- [Flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Caltech 101](https://www.tensorflow.org/datasets/catalog/caltech101)
- [MINC](http://opensurfaces.cs.cornell.edu/publications/minc/) (this may be a more challenging dataset)
- For a very extensive evaluation, the [VTAB](https://github.com/google-research/task_adaptation) Natural dataset group could be used. Although in my case, this might require modifying each dataset to have fewer instances per class.


### Results on Oxford Pets

Note: except for a change in optimizer, I'm using the default arguments. E.g. training duration and corresponding learning rate schedule are not tuned.

Initial results are mostly negative: Even though conditional batchnorm can slightly improve finetuning performance compared to only training the final classification layer, it does not come close to the performance of training end-to-end.

#### End-to-end finetuning

```
10 images per class:
	Resnet18:
		Best metric: 60.80675941814548 (epoch 46)
3 images per class: 
	Resnet18:
		Best metric: 45.680021801187216 (epoch 46)
1 image per class (batch size 16):
	Resnet18:
		Best metric: 21.31370945761788 (epoch 45)
	Resnet50:
		Best metric: 40.556009811937855 (epoch 48)
```


#### Last layer only (fc)

```
1 image per class (batch size 16):
	Resnet18:
		Best metric: 10.302534750613246 (epoch 44)
	Resnet50:
		Best metric: 29.272281275551922 (epoch 37)
```


#### Conditional batchnorm + fc

```
1 image per class (batch size 16):
	Resnet18:
		Best metric: 10.602343962932679 (epoch 44)
	Resnet50:
		Best metric: 29.463068956118832 (epoch 37)
```

### Results on Flowers 102

Results on Flowers 102 show a similar negative pattern:

#### End-to-end finetuning

```
1 image per class (batch size 16):
	Resnet18:
		Best metric: 22.94117648554783 (epoch 49)
```

#### Conditional batchnorm + fc

```
1 image per class (batch size 16):
	Resnet18:
		Best metric: 7.64705883100921 (epoch 40)
```

### TODO

- [x] find a suitable dataset
- [x] add EDA notebook for visualizing the data
- [x] modify dataset loading
- [x] modify model loading
- [x] Try a harder dataset than Oxford Pets where the risk of overfitting is higher
- [ ] look at the argument options for `train.py` and try to find better default baseline arguments. Maybe look at some reference implementations or papers for this. Note e.g. that lr scheduling is on by default (both upscaling and downscaling of lr)
- [ ] make it possible to plot the results in a graph for more easily comparing different runs. E.g. there is a summary.csv file which may be useful


### References

[1] De Vries et al., ["Modulating early visual processing by language"](https://arxiv.org/abs/1707.00683), 2017.