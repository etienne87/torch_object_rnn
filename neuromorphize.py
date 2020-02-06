import torch
import cv2
import numpy as np 
from core.utils.opts import cuda_time, time_to_batch, batch_to_time


@cuda_time
def neuromorphize(tensor, noise, threshold=0):
    # very naive & fast simulator of dvs
    # tensor T, N, C, H, W
    # first move T at the end & apply a conv2d to compute difference
    diff = tensor[1:] - tensor[:-1]
    n,c,h,w = diff.shape

    diff, _ = time_to_batch(diff)
    diff = batch_to_time(diff, tensor.shape[1])

    on = diff > threshold
    off = diff < threshold
    zero = diff.abs() < threshold

    diff[zero] = 127
    diff[on] = 255
    diff[off] = 0

    #diff *= (torch.rand(n,c,h,w)>0.1).to(diff)
    
    return diff*noise


def neuromorphize_video(video_filename, tbins=10, height=480, width=640):
    def read(cap):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height), 0, 0, cv2.INTER_AREA)
        if ret:
            frame = np.moveaxis(frame, 2, 0)
            frame = torch.from_numpy(frame)[None]
        return ret, frame

    cap = cv2.VideoCapture(video_filename)

    _, frame = read(cap)
    volume = [frame]

    _,c,h,w = frame.shape
    noise = (torch.rand(tbins,c,h,w)>0.1).cuda() 

    while cap:
        volume = [volume[-1]]
        for t in range(tbins):
            ret, frame = read(cap)
            if ret:
                volume.append(frame)
            else:
                break
        tensor = torch.cat(volume).cuda()
        neuromorphize(tensor, noise)


        # viz
        if not ret:
            break
        





if __name__ == '__main__':
    import fire
    fire.Fire(neuromorphize_video)
