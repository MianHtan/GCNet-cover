# GCNet for satellite
This is a non-official pytorch implementation of stereo matching network <a href="https://arxiv.org/abs/1703.04309">GCNet</a> and make a slight modification to adapt it to satellite image
### Differences from original paper
- In order to make the model applicable to satellite images, the option of minimum disparity was added to adapt it to negative disparity
- Using OneCycleLR to adjust the learning rate

### Dataset
The dataloader only support `DFC2019` and `WHU-Stereo` dataset, the dataset need to be preprocessed to 8 bits <br />
You can train the model by using `train.py`

```
python train.py
```
You can use tensorboard to monitoring the loss and learning rate during training.

```
tensorboard --logdir ./logs
```

### Environment
- torch                     2.1.1
- torchvision               0.16.1
- numpy                     1.24.1
- matplotlib                3.2.2
- opencv-python             4.8.1.78

### demo
You can test a single pair of stereo image using the notebook `demo.ipynb`