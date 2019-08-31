import numpy as np
from torchvision import datasets, transforms
import moving_box_detection as toy
import cv2

TRAIN_DATASET = datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

TEST_DATASET = datasets.MNIST('../data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))


class MovingMnistAnimation(toy.Animation):
    def __init__(self, t=10, h=128, w=128, c=1, max_stop=15,
                max_objects=3, train=True):
        self.dataset_ = TRAIN_DATASET if train else TEST_DATASET
        super(MovingMnistAnimation, self).__init__(t, h, w, c, max_stop, 'none', 10, max_objects, True)


    def reset(self):
        super(MovingMnistAnimation, self).reset()
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(self.dataset_))
            x, y = self.dataset_[idx]
            self.objects[i].class_id = y
            img = x.numpy()[0]
            abs_img = np.abs(img)
            y, x = np.where(abs_img > 0.5)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)
            self.objects[i].img = np.repeat(img[y1:y2, x1:x2][...,None], self.channels, 2)

    def run(self):
        self.img[...] = 0

        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, object in enumerate(self.objects):
            x1, y1, x2, y2 = object.run()
            boxes[i] = np.array([x1, y1, x2, y2, object.class_id])
            #draw in roi resized version of img
            thumbnail = cv2.resize(object.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail)

        output = np.transpose(self.img, [2, 0, 1])
        return output, boxes


class MovingMnistDataset(toy.SquaresVideos):
    def __init__(self, batchsize=32, t=10, h=300, w=300, c=3,
                 normalize=False, max_stops=30, max_objects=3,
                 max_classes=3, train=True):
        super(MovingMnistDataset, self).__init__(batchsize, t, h, w, c,
                                                normalize, max_stops, max_objects,
                                                max_classes, 'none', True)
        self.labelmap = [str(i) for i in range(10)]

    def reset(self):
        self.animations = [MovingMnistAnimation(self.time, self.height,
                                                self.width, self.channels, self.max_stops,
                                                 self.max_objects,
                                                 self.max_classes)
                           for _ in range(self.batchsize)]




if __name__ == '__main__':
    import torch
    from core.utils import boxarray_to_boxes, draw_bboxes, make_single_channel_display

    # anim = MovingMnistAnimation()
    # labelmap = [str(i) for i in range(10)]
    #
    # while 1:
    #     img, boxes = anim.run()
    #
    #     img = img[0][..., None]
    #     img = np.concatenate([img, img, img], axis=2)
    #
    #     boxes = torch.from_numpy(boxes)
    #     bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4], labelmap)
    #
    #     img = draw_bboxes(img, bboxes)
    #
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)

    dataloader = MovingMnistDataset(t=10, c=3, h=256, w=256, batchsize=32)

    start = 0
    for i, (x, y) in enumerate(dataloader):

        for j in range(1):
            for t in range(len(y)):
                boxes = y[t][j].cpu()
                bboxes = boxarray_to_boxes(boxes[:, :4], boxes[:, 4], dataloader.labelmap)

                if dataloader.render:
                    img = x[t, j, :].numpy().astype(np.float32)
                    if img.shape[0] == 1:
                        img = make_single_channel_display(img[0], -1, 1)
                    else:
                        img = np.moveaxis(img, 0, 2)
                        show = np.zeros((dataloader.height, dataloader.width, 3), dtype=np.float32)
                        show[...] = img
                        img = show
                else:
                    img = np.zeros((256, 256, 3), dtype=np.uint8)

                img = draw_bboxes(img, bboxes)

                cv2.imshow('example', img)
                key = cv2.waitKey(0)
                if key == 27:
                    exit()
