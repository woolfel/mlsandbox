# Fashion MNIST

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is a good testbench for trying out different NN configurations and hyper parameters. The current setup consistenty achieves 94% test accuracy and 98% training accuracy.

![accuracy graph](./model_accuracy.png)
![loss graph](./model_loss.png)
![training log](./training_log.png)


### fashion_trainer.py Model Configuration ###

|Layer (type)                |Output Shape            |Param  |
|----------------------------|------------------------|-------|
|conv2d (Conv2D)             |(None, 27, 27, 196)     |980    |
|conv2d_1 (Conv2D)           |(None, 26, 26, 256)     |200960 |
|max_pooling2d (MaxPooling2D)|(None, 13, 13, 256)     |0      |
|conv2d_2 (Conv2D)           |(None, 13, 13, 256)     |65792  |
|conv2d_3 (Conv2D)           |(None, 12, 12, 512)     |524800 |
|dropout (Dropout)           |(None, 12, 12, 512)     |0      |
|flatten (Flatten)           |(None, 73728)           |0      |
|dense (Dense)               |(None, 128)             |9437312|
|dropout_1 (Dropout)         |(None, 128)             |0      |
|dense_1 (Dense)             |(None, 10)              |1290   |
|Total params: 10,231,134                                     |
|Trainable params: 10,231,134                                 |
|Non-trainable params: 0                                      |

