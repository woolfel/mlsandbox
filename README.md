# MLSandbox

This is a sandbox for various ML experiments. It's for educational purposes and for fun.

### Software ###

* Tensorflow 1.10 or newer
* Keras 2.1.6
* Python 3.6.5 or newer
* Numpy
* Scikit

## Getting started

1. clone the repository

2. run the training script to produce some checkpoint files. Warning if you don't have a tensorflow setup with GPU it will take longer to produce checkpoint files. Example python ./cifar10_train3.py checkpoints/training1

3. After the training is done, there will be 22 files in the checkpoints/training1 folder

4. run diff script to compare two checkpoint files. Example python .\checkpoint_diff.py .\checkpoints\training3\weights.17-0.786-0.638.h5 .\checkpoints\training3\weights.22-0.790-0.663.h5 model_results.json

5. it will print out a summary and save the results to a file

## Calculating model diff between checkpoints

In cifar folder are some scripts for creating diff between two checkpoint models. Important note on the experiment and purpose.

1. the simple sequential model isn't going to produce useful model you can deploy in a real product. The model is small and simple to keep training time short.

2. the scripts are set to train for 22 epochs and tends to overfit to the training set. This is on purpose. I want to compare different checkpoints to understand why the weights ended up overfitting to the training set.

3. the training scripts will write checkpoint files to a folder with the validation accuracy and loss in the filename. this makes it easier to see which epoch had a degredation in accuracy and overfit.

4. there's no way to view the layer diff at the moment. the eventual goal is to render the weights, channels and filters difference in a graph.

5. the diff is saved to a file in json format. I am using marshmallow framework to write the data. The past argument should be the output filename
