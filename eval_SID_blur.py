import os
import argparse
from torchvision.transforms import Compose, ToTensor
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


def eval(model, testing_data_loader, model_path, output_folder):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    print('Evaluation: ', output_folder)
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
            
        input = input.cuda()
        
        with torch.no_grad():
            output = model(input)
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(), 0, 1)
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':

    net = CIDNet().cuda()

    import time
    start_time = time.time()

    for index in range(1, 230):
        test_dir = "./datasets/Sony_total_dark/test/short/"
        fill_index = '1' + str(index).zfill(4)
        now_dir = test_dir + fill_index + "/"
        model_path = "./weights/SID.pth"
        SID_folder = "./output/SID/"
        if not os.path.exists(SID_folder):
            os.mkdir(SID_folder)
        if os.path.exists(now_dir):
            output_folder = SID_folder + fill_index + "/"
            eval_data = DataLoader(dataset=get_eval_set(now_dir), num_workers=0, batch_size=1, shuffle=False)
            eval(net, eval_data, model_path, output_folder)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")