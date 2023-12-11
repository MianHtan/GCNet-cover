import torch
from GCNet.GCNet import GCNet

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

net = GCNet(device)
net = net.to(device)
x_l = torch.rand(2,3,512,512).to(device)
x_r = torch.rand(2,3,512,512).to(device)

disp = net(x_l, x_r, -64, 64)
print(disp.shape)