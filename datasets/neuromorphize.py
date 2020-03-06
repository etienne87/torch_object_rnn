import os
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np 
from core.utils.opts import cuda_tick
import pafy


def download_video(url, filepath="/tmp/"):
    """
    code to download a video from youtube
    :param url:
    :return:
    """
    video = pafy.new(url)
    best = video.getbest()
    best.download(quiet=False)
    filename = best.download(filepath=filepath)
    return filename


class Neuromorphizer(nn.Module):
    def __init__(self, height, width, video_fps, 
                    refractory_period_us=0, threshold=0, 
                    p_fix_pattern_noise=0.001, max_period_noise=10, max_tbins=100):
        super(Neuromorphizer, self).__init__()
        self.height = height
        self.width = width
        self.video_fps = video_fps
        self.delta_t_us = 1e6/video_fps
        self.register_buffer('state', torch.zeros((height,width), dtype=torch.short))
        self.register_buffer('timesurface', torch.zeros((height,width), dtype=torch.short))

        self.refractory_period_us = refractory_period_us
        print('ref: ', self.refractory_period_us, ' delta_t: ', self.delta_t_us)
        self.threshold = threshold 
        self.t_us = 0

        self.p_fix_pattern_noise = p_fix_pattern_noise
        self.register_buffer('on_noise', torch.rand(max_period_noise,height,width)<p_fix_pattern_noise)
        self.register_buffer('off_noise', torch.rand(max_period_noise,height,width)<p_fix_pattern_noise)
 
        self.register_buffer('diffs', torch.zeros((max_tbins, height,width), dtype=torch.uint8))
        

    def reset(self):
        self.state[...] = 0
        self.timesurface[...] = 0

    def forward(self, tensor):
        for i, i_t in enumerate(tensor):
            self.t_us += self.delta_t_us
            cnt = int(self.t_us // self.delta_t_us)  

            diff = (i_t - self.state) 

            if self.refractory_period_us >= self.delta_t_us:
                min_time = (self.t_us - self.refractory_period_us)//self.delta_t_us
                ready = self.timesurface <= min_time
                diff = diff * ready

            on = (diff > self.threshold)  
            off = diff < self.threshold
            zero = diff.abs() <= self.threshold

            non_zero = ~zero

            self.diffs[i][on] = 255
            self.diffs[i][off] = 0
            self.diffs[i][zero] = 127  

            self.state[non_zero] = i_t[non_zero]

            self.timesurface[non_zero] = cnt

            noise_on = self.on_noise[cnt%len(self.on_noise)]
            noise_off = self.off_noise[cnt%len(self.off_noise)]


            self.diffs[i][noise_on] = 255
            self.diffs[i][noise_off] = 0

        return self.diffs

  
class CvFramePipeline(object):
    def __init__(self, video_filename, height, width, seek_frame=0):
        self.height = height
        self.width = width
        self.cap = cv2.VideoCapture(video_filename)
        self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, seek_frame)

    def __iter__(self):
        while self.cap:
            ret, frame = self.cap.read()   
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.width, self.height), 0, 0, cv2.INTER_AREA)
                frame = torch.from_numpy(frame)
            yield ret, frame

class TensorPipeline(object):
    def __init__(self, tbins, frame_pipeline, cuda=True):
        self.tbins = tbins
        self.frame_pipe = frame_pipeline
        self.cuda = cuda
    
    def __iter__(self):
        volume = []
        for ret, frame in self.frame_pipe:
            if not ret:
                break
            volume.append(frame[None])
            if len(volume) == self.tbins:
                y = torch.cat(volume)
                if self.cuda:
                    y = y.cuda()
                y = y.short()
                volume = []
                yield y


def neuromorphize(tensor_pipeline, pix2nvs, viz):
    for tensor in tensor_pipeline:
        start = cuda_tick()
        tensor = pix2nvs(tensor)

        end = cuda_tick()
        rt = end-start
        freq = (1./rt) * len(tensor)
        print(freq, ' img/s')

        if viz:
            data = tensor.cpu().numpy()
            for img in data:
                im = img.astype(np.uint8)
                cv2.imshow('img', im)
                key = cv2.waitKey(5)
                if key == 27:
                    return

def neuromorphize_video(video_filename, threshold=5, tbins=120, height=480, width=640, seek_frame=0, ref=0, scene_fps=1000, p=0.001, viz=True):
    """Example of usage of neuromorphizer

    Take a OpenCV video & turn it into events
    """
    if os.path.isdir(video_filename):
        video_filenames = glob.glob(video_filename + '/*')
    else:
        video_filenames = [video_filename]
    
    pix2nvs = Neuromorphizer(height, width, scene_fps, 
                                refractory_period_us=ref*1e6/scene_fps, 
                                p_fix_pattern_noise=p,
                                threshold=threshold, 
                                max_period_noise=10,
                                max_tbins=tbins)
    pix2nvs.cuda()
    
    for video_filename in video_filenames:
        pix2nvs.reset()

        print('video: ', video_filename)

        frame_pipeline = CvFramePipeline(video_filename, height, width, seek_frame)
        tensor_pipeline = TensorPipeline(tbins, frame_pipeline)
        neuromorphize(tensor_pipeline, pix2nvs, viz)

    
if __name__ == '__main__':
    import fire
    fire.Fire(neuromorphize_video)
