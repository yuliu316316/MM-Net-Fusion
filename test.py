import numpy as np
import torch
from tqdm import tqdm
import os

from net.fusionnet import FusionNet
from dataset_spect_mri import DatasetMed
from config import config
from utils.utils_image import imsave
from torch.utils.data import DataLoader

def fusion():

    model=FusionNet()
    model_path = "model_path"

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        model.cuda()
        model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset_test = DatasetMed(phase='test',opt=config,transform=None)

    test_loader = DataLoader(dataset_test,
                              batch_size=1)
    predict_process = tqdm(test_loader)
    for i, inputs in enumerate(predict_process):
        input_A = inputs['A']
        input_B = inputs['B']
        imgname = inputs['A_path'][0]
        input_A = input_A.cuda()
        input_B = input_B.cuda()
        out = model(input_A,input_B)
        ones = torch.ones([1, 1, 256, 256]) # out_put size
        ones = torch.FloatTensor(ones)
        ones = ones.cuda()
        fusion_y = torch.mul(out, input_A) + torch.mul((ones - out), input_B)

        output = np.squeeze(fusion_y.detach().cpu().numpy())
        result = (output* 255).astype(np.uint8)
        save_name = os.path.join('output_path', os.path.basename(imgname))
        imsave(result, save_name)

if __name__ == '__main__':
    torch.cuda.set_device(1)
    fusion()
