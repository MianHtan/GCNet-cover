# GCNet for satellite
This is a non-official pytorch implementation of stereo matching network GCNet <https://arxiv.org/abs/1703.04309>
### Differences from original paper
- In order to make the model applicable to satellite images, the option of minimum disparity was added to adapt it to negative disparity
- Add code to train on DFC2019 dataset

### Environment
- torch                     2.1.1
- torchvision               0.16.1
- numpy                     1.24.1
- matplotlib                3.2.2
- opencv-python             4.8.1.78

### demo
You can test a single pair of stereo image using the notebook `demo.ipynb`