# UNet
Creating a UNet Convolutional Neural Network for interactively colourising black and white photos.

## Create Python virtual environment
```python3 - m venv .venv```

```source .venv/bin/activate```

```pip install -r requirements.txt```

## Paper
[Paper](paper/1505.04597v1.pdf)

[Uni-Freiberg: Olaf Ronneberger: 18 May 2015](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

[Debugger Cafe: Sovit Ranjan Rath: 3 April 2023](https://debuggercafe.com/unet-from-scratch-using-pytorch/)

Modified to take greyscale photos as input, then output LUV colorised photos


----
# Version 1
- No BatchNorm
- Image input and targets Normalized: $\mu$: 0.5, $\sigma$ 0.25 
- Epochs: 5
- Mean Squared Error Loss
- No dropout
- Learning Rate: 5e-5

## Training
Training loss (smoothed) / Time
![Training loss](<runs/old/Pasted image.png>)

## Results
![Results](examples/old/_combined_img.png)


Input | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Ground Truth 
----- | ----- | ----- | ----- | ----- | ----- | ----- | 


![Evaluation|300px](examples/old/eval_fig.png)


------

# TODO
- [x] Fix HSV normalization
- [x] Run performance evaluation
- [ ] Pytorch and Rust GUI : https://medium.com/@heyamit10/loading-and-running-a-pytorch-model-in-rust-f10d2577d570
- [ ] Dropout

-----

# Version 2 
Highly experimental trails of different error functions, normalisation methods, droput / no dropout
## Proposal
 - [x] Removed Image input and targets Normalized: $\mu$: 0.5, $\sigma$ 0.25 
 - [x] No normalisation 
 - [x] Improved Loss function: MSE + SSIM
 - [x] BatchNorm on down conv, hints down conv, up conv, and hints up conv
 - [x] Skip Connection Self Convolution on level 2, 3, 4, 5
 - [X] Hintegration: Convolve hints, then integrate then convolve colour hints alongside down convolutions.
 - [ ] Dropout (not from school, from network)
 - [ ] Variable learning rate
 - [ ] Variable hints per epoch
 - [x] 51,086,771 total trainable parameters.

 ## Epoch 1
 Num points = 150
 
 -----

 # Version 3
 ## Proposal
 61,508,906 total parameters.
 - [x] Convert to LAB colorspace
 - [x] 2 Channel output
 - [ ] K-means global hints
 - [ ] Sobel-MSE(O,Y),
 - [x] Hubert(O,Y) loss functions: https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
 - [x] Dropout
 - [x] Remove MaxPooling, downsample instead
 - [x] Reimplement Hints skip connections

 ## Notes
 This model improved upon the last version by fully implementing a hints feed-forward network. This model passed the hints up the U-Net structure to higher complexity convolutional layers, allowing hint information to permeate the network at the highest levels. LAB colour-space was implemented in this version which is easier to train on than HSV colour-space. This version also removed the max-pooling layers, instead making down-convolutions that reduce channel size. The first 5 epochs were trained with a Huber loss function, the last 3 epochs were trained on MSError loss functions. This allowed the network to train easily on high-variance data, later MSError was used to train the network back to expected results after epoch 5. 

 ## Results
![Results](examples/ver3/_combined_img.png)

## Evaluation
![Evaluation](examples/ver3/eval_fig.png)