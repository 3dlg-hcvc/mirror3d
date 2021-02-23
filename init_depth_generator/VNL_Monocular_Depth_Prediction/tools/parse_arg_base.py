import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    # TODO add more option here
    def initialize(self, parser):

        # TODO
        parser.add_argument('--model_name',                type=str,   help='model name', default='vnl')
        #TODO
        parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
        #TODO
        parser.add_argument('--mesh_depth',                action='store_true',  help='using coco input format or not')
        # TODO
        parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json')
        # TODO
        parser.add_argument('--coco_train',                type=str,   help='coco json path', default='/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json')
        # TODO
        parser.add_argument('--coco_train_root',           type=str,   help='coco data root', 
            default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise")
        # TODO
        parser.add_argument('--coco_val_root',             type=str,   help='coco data root', 
            default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise")
        # TODO
        parser.add_argument('--coco_focal_len',            type=int,   help='nyu : 519', dest='f', default=519)
        # TODO 
        parser.add_argument('--depth_shift',               type=int,   help='nyu : 1000, m3d : 4000', default=1000) # 4000 for m3d
        # TODO if coda boom
        parser.add_argument('--input_height',              type=int,   help='input height', default=256) # 480
        # TODO  if coda boom
        parser.add_argument('--input_width',               type=int,   help='input width',  default=320) # 640
        # TODO
        parser.add_argument('--batch_size',                type=int,   help='batch size', dest='batchsize', default=2)
        # TODO
        parser.add_argument('--learning_rate',             type=float, help='initial learning rate', dest='lr', default=1e-4)
        # TODO
        parser.add_argument('--resume_checkpoint_path',    type=str,   help='path to a checkpoint to load', dest='load_ckpt', default='')
        # TODO
        parser.add_argument('--checkpoint_save_freq',      type=int,   help='Checkpoint saving frequency in global steps /iteration; nyu 5000; m3d 10000', dest='siter' , default=100)
        # TODO
        # Log and save
        parser.add_argument('--log_directory',             type=str,   help='training output folder', default='/project/3dlg-hcvc/jiaqit/output', dest='results_dir')
        # TODO
        parser.add_argument('--num_epochs',                type=int,   help='number of epochs', dest='epoch', default=1)
        parser.add_argument('--output_save_folder',        type=str,   help='output_main_folder only use during inference', default='/project/3dlg-hcvc/jiaqit/exp_result')


        parser.add_argument('--cfg_file', default='lib/configs/resnext101_32x4d_nyudv2_class',
                            help='Set model and dataset config files')
        parser.add_argument('--dataset', default='nyudv2', help='Path to images')
        parser.add_argument('--resume', action='store_true', help='Resume to train')
        parser.add_argument('--start_epoch', default=0, type=int, help='Set training epochs')
        parser.add_argument('--start_step', default=0, type=int, help='Set training steps')
        parser.add_argument('--thread', default=4, type=int, help='Thread for loading data')
        parser.add_argument('--use_tfboard', action='store_true', help='Tensorboard to log training info')
        # parser.add_argument('--results_dir', type=str, default='output', help='Output dir')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
