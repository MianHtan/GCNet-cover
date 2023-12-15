import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
from pathlib import Path

from utils.stereo_datasets import fetch_dataset
from GCNet.GCNet import GCNet

def train(net, dataset_name, batch_size, root, min_disp, max_disp, iters, init_lr, resize, device, save_frequency=None, require_validation=False, pretrain = None):
    print("Train on:", device)

    # define model
    net.to(device)
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain), strict=True)
        print("Finish loading pretrain model!")
    else:
        net._init_params()
        print("Model parameters has been random initialize!")
    net.train()

    # fetch data
    train_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                batch_size = batch_size, resize = resize, mode = 'training')
    
    steps_per_iter = train_loader.__len__()
    num_steps = steps_per_iter * iters    
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr*2, num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
        
    criterion = nn.SmoothL1Loss().to(device)



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

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            # code of validation (on working)

            if total_steps % save_frequency == (save_frequency - 1):
                print("--- save checkpoint ---")

                # save checkpoints
                save_path = Path('training_checkpoints/%dsteps_GCNet_%s.pth' % (total_steps + 1, dataset_name))
                torch.save(net.state_dict(), save_path)

                # load data 
                if require_validation:
                    print("--- start validation ---")
                    test_loader = fetch_dataset(dataset_name = dataset_name, root = '/home/lab1/datasets/DFC2019_track2_grayscale_8bit',
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
                        print(f"--- after {total_steps + 1} steps vaildation loss: {val_loss:.7f} ---")

                net.train()
            #     # logger.write_dict(results)
            #     net.train()
            print(f"step{total_steps} loss: {loss:.7f} lr: {scheduler.get_last_lr()[0]}")
            total_steps += 1

            if total_steps > num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 1000:
            cur_iter = int((total_steps-1)/steps_per_iter)
            save_path = Path('training_checkpoints/%d_epoch_%s.pth' % (cur_iter, 'GCNet_DFC2019'))
            torch.save(net.state_dict(), save_path)

    print("FINISHED TRAINING")

    PATH = 'training_checkpoints/GCNet_DFC2019.pth'
    torch.save(net.state_dict(), PATH)
    print("model has been save to path: ", PATH)

if __name__ == '__main__':
    Path("training_checkpoints").mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    net = GCNet()
    net._init_params()
    net = net.to(device)

    # training set keyword: 'DFC2019', 'WHUStereo'
    # '/home/lab1/datasets/DFC2019_track2_grayscale_8bit'
    # '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth'
    train(net=net, dataset_name='WHUStereo', root = '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth', 
          batch_size=1, min_disp=-32, max_disp=64, iters=5, init_lr=0.0002,
          resize = [1024,1024], save_frequency = 500, require_validation=False, 
          device=device, pretrain="training_checkpoints/2_epoch_GCNet_DFC2019.pth")