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
    def __init__(self, height, width, video_fps, refractory_period_us=0, threshold=0, p_fix_pattern_noise=0.001):
        super(Neuromorphizer, self).__init__()
        self.height = height
        self.width = width
        self.video_fps = video_fps
        self.delta_t_us = 1e6/video_fps
        self.register_buffer('state', torch.zeros((height,width), dtype=torch.short))
        self.register_buffer('timesurface', torch.zeros((height,width), dtype=torch.short))

        self.refractory_period = refractory_period_us
        print('ref: ', self.refractory_period, ' delta_t: ', self.delta_t_us)
        self.threshold = threshold 
        self.t_us = 0

        self.p_fix_pattern_noise = p_fix_pattern_noise
        self.register_buffer('on_noise', torch.rand(10,height,width)<p_fix_pattern_noise)
        self.register_buffer('off_noise', torch.rand(10,height,width)<p_fix_pattern_noise)
        

    def forward(self, tensor):
        diffs = []
        for i_t in tensor:
            self.t_us += self.delta_t_us
            cnt = int(self.t_us // self.delta_t_us)

            self.min_time = (self.t_us - self.refractory_period)//self.delta_t_us

            idle = self.timesurface <= self.min_time

            diff = (i_t - self.state) 
            diff = diff * idle

            on = (diff > self.threshold)  
            off = diff < self.threshold
            zero = diff.abs() <= self.threshold

            non_zero = ~zero
            diff[on] = 255
            diff[off] = 0
            diff[zero] = 127  
            self.state[non_zero] = i_t[non_zero]

            self.timesurface[non_zero] = cnt

            noise_on = self.on_noise[cnt%len(self.on_noise)]
            noise_off = self.off_noise[cnt%len(self.off_noise)]

            diff[noise_on] = 255
            diff[noise_off] = 0
            diffs.append(diff[None])

        diff = torch.cat(diffs)
        
        return diff


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

class ToTensorPipeline(object):
    def __init__(self, tbins, frame_pipeline, cuda=True):
        self.tbins = tbins
        self.frame_pipe = frame_pipeline
        self.cuda = cuda
    
    def __iter__(self):
        volume = []
        for ret, frame in self.frame_pipe:
            if not ret:
                continue
            volume.append(frame[None])
            if len(volume) == self.tbins:
                y = torch.cat(volume).short()
                if self.cuda:
                    y = y.cuda()
                volume = []
                yield y


def neuromorphize_video(video_filename, threshold=0, tbins=120, height=480, width=640, seek_frame=0, scene_fps=1000, p=0.001):
    frame_pipeline = CvFramePipeline(video_filename, height, width, seek_frame)

    tensor_pipeline = ToTensorPipeline(tbins, frame_pipeline)


    pix2nvs = Neuromorphizer(height, width, scene_fps, refractory_period_us=0*1e6/scene_fps, threshold=threshold)
    pix2nvs.cuda()
    

    for tensor in tensor_pipeline:

        start = cuda_tick()
        tensor = pix2nvs(tensor)

        end = cuda_tick()
        rt = end-start
        freq = (1./rt) * tbins
        print(freq, ' img/s')

        data = tensor.cpu().numpy()
        for img in data:
            im = img.astype(np.uint8)
            cv2.imshow('img', im)
            cv2.waitKey(0)

    


if __name__ == '__main__':
    import fire
    fire.Fire(neuromorphize_video)
