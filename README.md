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