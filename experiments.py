import os
from os.path import join as pj
import shutil

from datetime import datetime

import json


class Experiment(object):
    def __init__(self, root, name=''):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if name == '':
            self.name = '{}'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        else:
            self.name = '{}-{}'.format(name, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

        self.dir = pj(self.root, self.name)
        os.makedirs(self.dir)

        self.dirs = dict()

    def add_dir(self, name):
        new_dir = pj(self.dir, name)
        self.dirs[name] = new_dir

        os.makedirs(new_dir)

        return new_dir

    def save_file(self, file_path):
        shutil.copy(file_path, pj(self.dir, os.path.basename(file_path)))

    def save_json(self, name, data):
        with open(pj(self.dir, name), 'w') as fout:
            json.dump(data, fout)