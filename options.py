from argparse import ArgumentParser

class Options:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--mode', type=str, default='', help='train | generate | reveal | detect')
        self.parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
        self.parser.add_argument('--imageSize', type=int, default=256, help='the number of frames')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
        self.parser.add_argument('--beta_adam', type=float, default=0.5, help='beta_adam for adam. default=0.5')
        self.parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
        self.parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
        self.parser.add_argument('--Hnet_mode', type=str, default='UNetDeep', help='UNetDeep | UNetShallow')
        self.parser.add_argument('--Rnet_mode', type=str, default='FullConvSkip', help='FullConvSkip | FullConv | TransConv')
        self.parser.add_argument('--Rnet_beta', type=float, default=0.75, help='hyper parameter of Hnet factor')
        self.parser.add_argument('--Hnet_factor', type=float, default=1, help='hyper parameter of Hnet factor')
        self.parser.add_argument('--checkpoint', default='', help='checkpoint address')

        self.parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
        self.parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
        self.parser.add_argument('--norm', default='instance', help='batch or instance')
        self.parser.add_argument('--loss', default='l2', help='l1 or l2')
        self.parser.add_argument('--Rloss_mode', default='secret0', help='secret0 | secret1 | mask')
        self.parser.add_argument('--Hnet_inchannel', type=int, default=3, help='1: gray; 3: color')
        self.parser.add_argument('--Hnet_outchannel', type=int, default=3, help='1: gray; 3: color')
        self.parser.add_argument('--Rnet_inchannel', type=int, default=3, help='1: gray; 3: color')
        self.parser.add_argument('--Rnet_outchannel', type=int, default=3, help='1: gray; 3: color')
        self.parser.add_argument('--max_val_iters', type=int, default=200)
        self.parser.add_argument('--max_train_iters', type=int, default=2000)

        self.parser.add_argument('--bs_train', type=int, default=16, help='training batch size')
        self.parser.add_argument('--bs_generate', type=int, default=16, help='generation batch size')
        self.parser.add_argument('--bs_extract', type=int, default=16, help='extraction batch size')

        self.parser.add_argument('--output_dir', default='', help='directory of outputing results')
        self.parser.add_argument('--val_dir', type=str, default='', help='directory of validation images in training process')
        self.parser.add_argument('--train_dir', type=str, default='', help='directory of training images in training process')
        self.parser.add_argument('--cover_dir', type=str, default='', help='directory of cover images')
        self.parser.add_argument('--container_dir', type=str, default='', help='directory of container images')
        self.parser.add_argument('--secret_dir', type=str, default='', help='directory of secret images')
        self.parser.add_argument('--rev_secret_dir', type=str, default='', help='directory of revealed secret images')

        self.parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
        self.parser.add_argument('--dis_num', type=int, default=5, help='number of example image for visualization')
        self.parser.add_argument('--threshold', type=float, default=0.9, help='value to decide whether a pixel is tampered')

        self.parser.add_argument('--gen_mode', type=str, default='white', help='white | random | same')

        self.parser.add_argument('--secret_mode', type=str, default='RGB', help='RGB | Gray | QRCode')

        self.parser.add_argument('--mask_mode', type=str, default='random', help='random | block | none | mixed')
        self.parser.add_argument('--block_size', type=int, default=32, help='bigger block size correspond to smaller masked block')
        self.parser.add_argument('--block_ratio', type=float, default=0.5, help='')

        self.parser.add_argument('--attack', default='Yes', type=str)
        self.parser.add_argument('--jpeg_quality', default=50, type=int)
        self.parser.add_argument('--gaussian_kernelSize', default=5, nargs='+', type=int)
        self.parser.add_argument('--gaussian_sigma', default=0.1, nargs='+',  type=float)  

    def parse(self):
        opts = self.parser.parse_args()
        return opts      