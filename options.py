import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)



        self.parser.add_argument('--num_frames', type=int, default=60000, help= '# of frames for use,3680')
        self.parser.add_argument('--device', default='cuda', help='cuda:[d] | cpu')
        self.parser.add_argument('--seed',  type=int, default=17, help='random seed')
        self.parser.add_argument('--is_rgb', default=False, help='# if the images are RGB')
        self.parser.add_argument('--is_normal', default=True, help='# if the images are RGB')
        self.parser.add_argument('--is_inverse', default=True, help='# if the images are RGB')

        #directory
        # self.parser.add_argument('--base_data_dir', default='./combined_data_0903_off_knife', help='directory for data')
        self.parser.add_argument('--base_tactile1_dir', default='tactile1', help='directory for tactile1 data')
        self.parser.add_argument('--base_tactile2_dir', default='tactile2', help='directory for tactile2 data')
        self.parser.add_argument('--base_force_dir', default='force', help='directory for force data')
        self.parser.add_argument('--base_tactile1_ref_dir', default='tactile1_frame_ref.jpg', help='directory for tactile2 data')
        self.parser.add_argument('--base_tactile2_ref_dir', default='tactile2_frame_ref.jpg', help='directory for tactile2 data')


        self.parser.add_argument('--save_root', default='./save/', help='directory for saving ')
        self.parser.add_argument('--save_images_dir', default='images', help='directory for saving images')
        self.parser.add_argument('--save_model_dir', default='model', help='directory for saving models')
        # self.parser.add_argument('--save_features_dir', default='feature', help='directory for saving features')
        self.parser.add_argument('--save_plot', default='loss',help='directory for plotting loss')
        self.parser.add_argument('--load_model', default='net_best_loss.pth', help='directory for plotting loss')


        # dataset setting


        # training details
        self.parser.add_argument('--print_interval', type=int, default=1, help='# iterations to output loss')
        self.parser.add_argument('--schedsamp_k', type=float, default=900, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling,default 900.')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
        self.parser.add_argument('--batch_size_test', type=int, default=1, help='batch size for testing')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='the base learning rate of the generator')
        self.parser.add_argument('--epochs', type=int, default=300, help='# total training epoch')
        self.parser.add_argument('--patience', type=int, default=40, help='# patience for early stopping')
        self.opt = None




    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        # if not os.path.exists(self.opt.output_dir):
        #     os.makedirs(self.opt.output_dir)
        return self.opt

