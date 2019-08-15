import numpy as np
from torchvision import datasets, transforms
import toy_pbm_detection as toy
import cv2


class MovingMnistDataset(toy.Animation):
    def __init__(self, t=10, h=300, w=300, c=1, max_stop=15,
                max_objects=3, train=True):
        self.dataset_ = datasets.MNIST('../data', train=train, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

        # t, h, w, c, max_stop, 'none', max_classes = 1, max_objects = 3, render = True
        super(MovingMnistDataset, self).__init__(t, h, w, c, max_stop, 'none', 10, max_objects, True)


    def reset(self):
        super(MovingMnistDataset, self).reset()
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(self.dataset_))
            x, y = self.dataset_[idx]
            self.objects[i].class_id = y

            img = x.numpy()[0]

            abs_img = np.abs(img)

            y, x = np.where(abs_img > 0.5)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)

            self.objects[i].img = img[y1:y2, x1:x2]


    def run(self):
        self.img[...] = 0

        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()
            boxes[i] = np.array([x1, y1, x2, y2, object.class_id])

            #draw in roi resized version of img
            thumbnail = cv2.resize(object.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2, 0] = np.maximum(self.img[y1:y2, x1:x2, 0], thumbnail)


        output = np.transpose(self.img, [2, 0, 1])

        return output, boxes

if __name__ == '__main__':
    import torch
    from core.utils import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    anim = MovingMnistDataset()
    labelmap = [str(i) for i in range(10)]

    while 1:
        img, boxes = anim.run()


        img = img[0][..., None]
        img = np.concatenate([img, img, img], axis=2)

        boxes = torch.from_numpy(boxes)
        bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4], labelmap)

        img = draw_bboxes(img, bboxes)

        cv2.imshow('img', img)
        cv2.waitKey(0)



