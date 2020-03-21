import logging
import time
from threading import Thread

import cv2
import numpy as np
from worker.state import State
from worker.video_reader import VideoReader
from worker.video_writer import VideoWriter
from cnd.ocr.predictor import Predictor
from datetime import datetime

deltas = [-55, -36, -20, -4, 14, 29]


class Visualizer:
    def _draw_ocr_text(self, frame, text):
        if text:
            art = np.resize(frame, (frame.shape[0] + 30, frame.shape[1], frame.shape[2]))
            art[-30:] = 0
            mid = (frame.shape[1] // 2, frame.shape[0] - 15)
            font = cv2.FONT_HERSHEY_SIMPLEX

            def draw_char(img, ch, num):
                if ch.isdigit():
                    size = 0.8
                else:
                    size = 0.6
                std = (mid[0] + deltas[num] + 5, mid[1] + 10)
                cv2.putText(img, ch, std, font, size, (0, 0, 0), thickness=2)

            frame = cv2.rectangle(frame, (mid[0] - 55, mid[1] - 12), (mid[0] + 55, mid[1] + 12), (255, 255, 255),
                                thickness=-1)

            for i in range(len(text)):
                draw_char(frame, text[i], i)

        return frame

    def __call__(self, frame, text):
        frame = self._draw_ocr_text(frame, text)
        return frame


class VisualizeStream:
    def __init__(self, name,
                 in_video: VideoReader,
                 state: State, video_path, fps, frame_size, coord, answer):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.coord = coord
        self.fps = fps
        self.frame_size = tuple(frame_size)

        self.out_video = VideoWriter("VideoWriter", video_path, self.fps, self.frame_size)
        self.sleep_time_vis = 1. / self.fps
        self.in_video = in_video
        self.stopped = True
        self.visualize_thread = None

        self.predictor = Predictor()

        self.visualizer = Visualizer()

        self.answer = answer
        self.best_accuracy = 0
        self.best_frame = -1
        self.best_result = ''

        self.logger.info("Create VisualizeStream")

    def _visualize(self):
        try:

            begin = datetime.now()
            frame_num = 0
            while True:
                if self.stopped:
                    return

                frame = self.in_video.read()
                if frame is None:
                    self.state.exit_event.set()
                    end = datetime.now()
                    print('FRAMES: ', frame_num,  end - begin)
                    return
                prediction = self.predictor.predict(frame)

                hits = 0
                for i in range(6):
                    if len(prediction) > i and self.answer[i] == prediction[i]:
                        hits += 1

                if hits > self.best_accuracy:
                    self.best_accuracy = hits
                    self.best_result = prediction
                    self.best_frame = frame_num

                frame = self.visualizer(frame, prediction)
                self.out_video.write(frame)

                time.sleep(self.sleep_time_vis)

                frame_num += 1

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def start(self):
        self.logger.info("Start VisualizeStream")
        self.stopped = False
        self.visualize_thread = Thread(target=self._visualize, args=())
        self.visualize_thread.start()
        self.in_video.start()

    def stop(self):
        self.logger.info("Stop VisualizeStream")
        self.stopped = True
        self.out_video.stop()
        if self.visualize_thread is not None:
            self.visualize_thread.join()
