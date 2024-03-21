import torch.utils.data as data
import utils.utils_image as util
from config import config
from PIL import Image
import torchvision.transforms as transforms


class DatasetMed(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, phase, opt,transform=None):
        super(DatasetMed, self).__init__()
        print('Dataset:  Harvard  Multi-modal Image Fusion.')
        self.phase = phase
        self.opt = opt
        self.transform = transform
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3

        # ------------------------------------
        if phase == "train":
            self.paths_A = util.get_image_paths('')
            self.paths_B = util.get_image_paths('')

        else:
            self.paths_A = util.get_image_paths('/media/liuyu/yuchen/code/testdata/mri')
            self.paths_B = util.get_image_paths('/media/liuyu/yuchen/code/testdata/spect')

    def __getitem__(self, index):

        if self.phase == 'train':
            """
            # --------------------------------
            # get under/over/norm patch pairs
            # --------------------------------
            """
            A_path = self.paths_A[index]
            B_path = self.paths_B[index]
            img_A = Image.open(A_path).convert('L')
            img_B = Image.open(B_path).convert('L')
            tran = transforms.ToTensor()
            img_A = tran(img_A)
            img_B = tran(img_B)

        else:  # test/ dataset
            """
            # --------------------------------
            # get under/over/norm image pairs
            # --------------------------------
            """
            A_path = self.paths_A[index]
            B_path = self.paths_B[index]
            img_A = Image.open(A_path).convert('L')
            img_B = Image.open(B_path).convert('L')
            tran = transforms.ToTensor()  # /255
            img_A = tran(img_A)
            img_B = tran(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)


if __name__ == '__main__':
    a = DatasetMed(phase="test", opt=config,transform=None)
    print(len(a))

