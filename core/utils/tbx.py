from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorboardX.summary import _clean_tag
from tensorboardX.proto.summary_pb2 import Summary
import imageio
import tempfile


def prepare_ckpt_dir(filename):
    dir = os.path.dirname(filename)
    if not os.path.isdir(dir):
        os.mkdir(dir)


def make_video(tensor, fps):
    t, h, w, c = tensor.shape
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name + '.gif'
    images = [tensor[t] for t in range(tensor.shape[0])]
    imageio.mimwrite(filename, images, duration=0.04)
    with open(filename, 'rb') as f:
        tensor_string = f.read()
    try:
        os.remove(filename)
    except OSError:
        pass

    return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=tensor_string)


def add_video(writer, tag, tensor_thwc, global_step=None, fps=30, walltime=None):
    """found that add_video from tbX is buggy"""
    tag = _clean_tag(tag)
    video = make_video(tensor_thwc, fps)
    summary = Summary(value=[Summary.Value(tag=tag, image=video)])
    writer.file_writer.add_summary(summary, global_step, walltime)


