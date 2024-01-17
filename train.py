import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path




from utils.stereo_datasets import fetch_dataset
from GCNet.GCNet import GCNet

def train(net, dataset_name, batch_size, root, min_disp, max_disp, iters, init_lr, resize, device, save_frequency=None, require_validation=False, pretrain = None):
    print("Train on:", device)
    Path("training_checkpoints").mkdir(exist_ok=True, parents=True)

    # tensorboard log file
    writer = SummaryWriter(log_dir='logs')

    # define model
    net.to(device)
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain), strict=True)
        print("Finish loading pretrain model!")
    else:
        net._init_params()
        print("Model parameters has been random initialize!")
    net.train()

    # fetch traning data
    train_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                batch_size = batch_size, resize = resize, mode = 'training')
    
    steps_per_iter = train_loader.__len__()
    num_steps = steps_per_iter * iters    
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    # initialize the optimizer and lr scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr*2, num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    criterion = nn.SmoothL1Loss().to(device)

    # start traning
    should_keep_training = True
    total_steps = 0
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
            valid = valid.detach_()

            net.training
            disp_predictions = net(image1, image2, min_disp, max_disp)
            assert net.training

            disp_predictions = disp_predictions.squeeze(1)
            loss = criterion(disp_predictions[valid], disp_gt[valid])
            loss.backward()
            optimizer.step()
            scheduler.step()

            # code of validation
            if total_steps % save_frequency == (save_frequency - 1):
                # save checkpoints
                save_path = Path('training_checkpoints/%dsteps_GCNet_%s.pth' % (total_steps + 1, dataset_name))
                torch.save(net.state_dict(), save_path)

                # load validation data 
                if require_validation:
                    print("--- start validation ---")
                    test_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                    batch_size = batch_size, resize = resize, mode = 'testing')
                    
                    net.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for i_batch, data_blob in enumerate(tqdm(test_loader)):
                            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
                            disp_predictions = net(image1, image2, min_disp, max_disp)
                            disp_predictions = disp_predictions.squeeze(1)
                            val_loss += criterion(disp_predictions[valid], disp_gt[valid])
                        val_loss = val_loss / (test_loader.__len__() * batch_size)
                        
                    writer.add_scalar(tag="vaildation loss", scalar_value=val_loss, global_step=total_steps+1)

                net.train()
            
            # write loss and lr to log
            writer.add_scalar(tag="training loss", scalar_value=loss, global_step=total_steps+1)
            writer.add_scalar(tag="lr", scalar_value=scheduler.get_last_lr()[0], global_step=total_steps+1)
            total_steps += 1

            if total_steps > num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 1000:
            cur_iter = int(total_steps/steps_per_iter)
            save_path = Path('training_checkpoints/%d_epoch_GCNet_%s.pth' % (cur_iter, dataset_name))
            torch.save(net.state_dict(), save_path)

    print("FINISHED TRAINING")

    final_outpath = f'training_checkpoints/GCNet_{dataset_name}.pth'
    torch.save(net.state_dict(), final_outpath)
    print("model has been save to path: ", final_outpath)

if __name__ == '__main__':
    

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    net = GCNet()

    # training set keywords: 'DFC2019', 'WHUStereo', 'all'
    # '/home/lab1/datasets/DFC2019_track2_grayscale_8bit'
    # '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth'
    train(net=net, dataset_name='DFC2019', root = '/home/lab1/datasets/DFC2019_track2_grayscale_8bit', 
          batch_size=1, min_disp=-32, max_disp=64, iters=3, init_lr=0.0001,
          resize = [1024,1024], save_frequency = 2000, require_validation=False, 
          device=device, pretrain="checkpoints/7_epoch_GCNet_DFC2019.pth")