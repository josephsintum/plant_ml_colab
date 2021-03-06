from time import time

import cv2
import numpy as np
import torch


class ObjectDetection:
    """
    The class performs generic object detection on a video.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    """

    def __init__(self):
        """
        :return: void
        """
        self.model = self.load_model()
        # self.model.conf = 0.4  # set inference threshold at 0.3
        # self.model.iou = 0.3  # set inference IOU threshold at 0.3
        # self.model.classes = [0]  # set model to only detect "one" class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_stream(self):
        """
        Function creates a streaming object to read the video from webcam.
        :param self:  class object
        :return:  OpenCV object to stream video frame by frame.
        """
        stream = cv2.VideoCapture(0)
        assert stream is not None
        return stream

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        path = 'plantdoc-yolov5.pth'
        model.load_state_dict(torch.load(path), strict=False)
        return model

    def score_frame(self, frame):
        """
        #     function scores each frame of the video and returns results.
        #     :param frame: frame to be inferred.
        #     :return: labels and coordinates of objects found.
        #     """
        self.model.to(self.device)
        results = self.model([frame])
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def plot_boxes(self, results, frame):
        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2:
                continue
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            bgr = (0, 255, 0)  # color of the box
            classes = self.model.names  # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
            # cv2.putText(frame, classes[labels[i]], (x1, y1), label_font, 0.9, bgr, 2)  # Put a label over box.

        return frame

    def __call__(self):
        stream = self.get_stream()  # Get your video stream.
        assert stream.isOpened()  # Verify that there is a stream.
        while True:
            start_time = time()  # We would like to measure the FPS.
            ret, frame = stream.read()  # Read the first frame.

            results = self.score_frame(frame)  # Score the Frame
            frame = self.plot_boxes(results, frame)  # Plot the boxes.

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {int(fps)}")
            if ret:
                # Display the resulting frame
                cv2.imshow('Plant Disease AI', frame)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        stream.release()
        cv2.destroyAllWindows()


a = ObjectDetection()
a()
