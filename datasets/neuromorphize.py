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


#TODO: simulate refr period?
#TODO: expand video rate with deep learning
def neuromorphize_sequence(tensor, state, threshold):
    diffs = []
    for i_t in tensor:
        diff = i_t - state
        on = diff > threshold
        off = diff < threshold
        zero = diff.abs() <= threshold
        diff[on] = 255
        diff[off] = 0
        diff[zero] = 127  
        state[~zero] = i_t[~zero]
        diffs.append(diff[None])
    diff = torch.cat(diffs)
    return diff, state


def neuromorphize_video(video_filename, threshold=0, tbins=120, height=480, width=640, p=0.01):
    def read(cap):
        ret, frame = cap.read()   
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (width, height), 0, 0, cv2.INTER_AREA)
            frame = torch.from_numpy(frame)
        return ret, frame

    cap = cv2.VideoCapture(video_filename)

    cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, int(1e4))

    _, frame = read(cap)

    state = frame.short().cuda()
    h,w = frame.shape

    #fix-pattern noise (hot pixels)
    if p > 0:
        on_noise = (torch.rand(tbins,h,w)<p).cuda() 
        off_noise = (torch.rand(tbins,h,w)<p).cuda()
    while cap:
        # volume = [volume[-1]]

        volume = []
        for t in range(tbins):
            ret, frame = read(cap)
            if ret:
                volume.append(frame[None])
            else:
                break
        tensor = torch.cat(volume).short().cuda()


        start = cuda_tick()
        tensor, state = neuromorphize_sequence(tensor, state, threshold)

        if p>0:
            tensor[on_noise] = 255
            tensor[off_noise] = 0

        end = cuda_tick()
        rt = end-start
        freq = (1./rt) * tbins
        print(freq, ' img/s')


        data = tensor.cpu().numpy()
        for img in data:
            im = img.astype(np.uint8)
            cv2.imshow('img', im)
            cv2.waitKey(0)
        # viz
        if not ret:
            break
    


if __name__ == '__main__':
    import fire
    fire.Fire(neuromorphize_video)
