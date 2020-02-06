import torch
import cv2
import numpy as np 
from core.utils.opts import cuda_tick, time_to_batch, batch_to_time
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
    zero = diff.abs() <= threshold

   
    diff[on] = 255
    diff[off] = 0
    diff[zero] = 127

    return diff #*noise


def neuromorphize_video(video_filename, threshold=0, tbins=60, height=480, width=640):
    def read(cap):
        ret, frame = cap.read()   
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (width, height), 0, 0, cv2.INTER_AREA)
            frame = torch.from_numpy(frame)[None,None]
        return ret, frame

    cap = cv2.VideoCapture(video_filename)

    cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, int(1e4))

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
        tensor = torch.cat(volume).int().cuda()


        start = cuda_tick()
        tensor = neuromorphize(tensor, noise, 6)
        end = cuda_tick()
        rt = end-start
        freq = (1./rt) * tbins
        print(freq, ' img/s')


        data = tensor.cpu().numpy()
        for img in data:
            im = img[0].astype(np.uint8)
            cv2.imshow('img', im)
            cv2.waitKey(1)
        # viz
        if not ret:
            break
    


if __name__ == '__main__':
    import fire
    fire.Fire(neuromorphize_video)
