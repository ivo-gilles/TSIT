import os
from data.pix2pix_dataset import Pix2pixDataset
from pathlib import Path


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
        scale_root = opt.scale_root
        c_image_paths, s_image_paths = [], []

        # Get Scale paths
        for where in ['Downtown', 'Highway', 'Rural', 'Sub-urban']:
            dir_path = os.path.join(opt.scale_root, "train_672w", "images", str(where))

            c_image_paths_read = os.listdir(dir_path + "/Clear")
            c_image_paths += [os.path.join(dir_path, "Clear", p) for p in c_image_paths_read if p != '']
            c_image_paths_read = os.listdir(dir_path + "/Cloudy")
            c_image_paths += [os.path.join(dir_path, "Cloudy", p) for p in c_image_paths_read if p != '']
            s_image_paths_read = os.listdir(dir_path + "/Night")
            s_image_paths += [os.path.join(dir_path, "Night", p) for p in s_image_paths_read if p != '']

        instance_paths = []

        length = min(len(c_image_paths), len(s_image_paths))
        print(f"Scale data: {len(c_image_paths)} day, {len(s_image_paths)} night, use {length} images each")
        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        # Get BDD100K path
        if opt.use_bdd100k:
            bdd_root = opt.bdd_root
            with open(os.path.join(bdd_root, 'bdd100k_lists/day2night/day_%s.txt' % opt.phase)) as c_list:
                c_image_paths_read = c_list.read().splitlines()
                c_image_paths += [os.path.join(bdd_root, p) for p in c_image_paths_read if p != '']
            with open(os.path.join(bdd_root, 'bdd100k_lists/day2night/night_%s.txt' % opt.phase)) as s_list:
                s_image_paths_read = s_list.read().splitlines()
                s_image_paths += [os.path.join(bdd_root, p) for p in s_image_paths_read if p != '']
            scale_length = length
            length = min(len(c_image_paths), len(s_image_paths))
            print(f"BDD100K data: {len(c_image_paths) - scale_length} day, {len(s_image_paths) - scale_length} night, use {length - scale_length} images each")
            c_image_paths = c_image_paths[:length]
            s_image_paths = s_image_paths[:length]

        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
