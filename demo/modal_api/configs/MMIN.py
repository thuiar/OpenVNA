import argparse
import torch
from modal_api import models
from modal_api.utils.functions import make_path

class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
        
        # model parameters
        parser.add_argument('--model', type=str, default='MMIN', help='chooses which model to use. [utt_fusion | mmin]')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--has_test', type=bool, default=False, help='has test [false,true].')
        parser.add_argument('--has_base', type=bool, default=True, help='input batch size 32 ,64')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='multimodal', help='chooses how datasets are loaded. [multimodal, multimodal_miss]')
        parser.add_argument('--num_workers', type=int, default=0, help='num workers of loading data')
        
        parser.add_argument('--datasetName', type=str, default='mosi', help='support mosi/mosei')
        parser.add_argument('--alignment', type=str, default='unaligned_v171_a25_l50', help='support unaligned_50/our_unaligned/MOSI')
        parser.add_argument('--augment', type=str, default='none', help='support none/method_one/method_two/method_three')
        parser.add_argument('--augment_rate', type=int, default=0.2, help='0.1, 0.2, 0.4')
        parser.add_argument('--test_mode', type=str, default='block_drop', help='support frame_drop/block_drop/random_drop')
        parser.add_argument('--test_seed_list', type=list, default=[1, 11, 111, 1111, 11111], help='indicates the seed for test period imperfect construction')
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        
        ## training parameter
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # expr setting 
        parser.add_argument('--gpu_ids', type=list, default=[0],  help='indicates the gpus will be used. If none, the most-free gpu will be used!')
        self.isTrain = True
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self, seed):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.seed = seed
        # process opt.suffix
        if opt.has_base:
            opt.name = f'{opt.datasetName}_{opt.model}_{opt.alignment}'
        else:
            opt.name = f'{opt.datasetName}_{opt.model}_{opt.alignment}_nobase'

        print("Expr Name:", opt.name)

        # set gpu ids
        str_ids = opt.gpu_ids
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            
        self.opt = opt
        return self.opt
