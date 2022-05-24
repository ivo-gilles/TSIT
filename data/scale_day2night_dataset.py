import os
from data.pix2pix_dataset import Pix2pixDataset


class ScaleDay2NightDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # image size is 672 x 384
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        scale_croot = opt.scale_croot
        scale_sroot = opt.scale_sroot
        c_image_paths, s_image_paths = [], []

        # Get Scale paths
        c_image_paths_read = os.listdir(scale_croot)
        c_image_paths += [os.path.join(scale_croot, p) for p in c_image_paths_read if p != '']
        s_image_paths_read = os.listdir(scale_sroot)
        s_image_paths += [os.path.join(scale_sroot, p) for p in s_image_paths_read if p != '']

        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        # Get BDD100K path
        if opt.use_bdd100k:
            bdd_croot = opt.bdd_croot
            bdd_sroot = opt.bdd_sroot
            with open(os.path.join(bdd_croot, 'bdd100k_lists/day2night/day_%s.txt' % opt.phase)) as c_list:
                c_image_paths_read = c_list.read().splitlines()
                c_image_paths += [os.path.join(bdd_croot, p) for p in c_image_paths_read if p != '']
            with open(os.path.join(bdd_sroot, 'bdd100k_lists/day2night/night_%s.txt' % opt.phase)) as s_list:
                s_image_paths_read = s_list.read().splitlines()
                s_image_paths += [os.path.join(bdd_sroot, p) for p in s_image_paths_read if p != '']
            length = min(len(c_image_paths), len(s_image_paths))
            c_image_paths = c_image_paths[:length]
            s_image_paths = s_image_paths[:length]

        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
