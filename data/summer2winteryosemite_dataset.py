import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class Summer2WinterYosemiteDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        croot = opt.croot
        sroot = opt.sroot

        try:
            c_image_dir = os.path.join(croot, '%sA' % opt.phase)
            c_image_paths = sorted(make_dataset(c_image_dir, recursive=True))
        except:
            # make sure the folder contains only image files (png, jpg, ...)
            c_image_paths_read = os.listdir(croot)
            c_image_paths = [os.path.join(croot, p) for p in c_image_paths_read if p != '']


        try:
            s_image_dir = os.path.join(sroot, '%sB' % opt.phase)
            s_image_paths = sorted(make_dataset(s_image_dir, recursive=True))
        except:
            # make sure the folder contains only image files (png, jpg, ...)
            s_image_paths_read = os.listdir(sroot)
            s_image_paths = [os.path.join(sroot, p) for p in s_image_paths_read if p != '']

        if opt.phase == 'train':
            s_image_paths = s_image_paths + s_image_paths

        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
