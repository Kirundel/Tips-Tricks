import cv2

cv2.setNumThreads(0)
import sys
import logging
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
# from worker.ocr_stream import OcrStream
from worker.visualize_stream import VisualizeStream
import json


def setup_logging(path, level='INFO'):
    handlers = [logging.StreamHandler()]
    file_path = path
    if file_path:
        file_handler = RotatingFileHandler(filename=file_path,
                                           maxBytes=10 * 10 * 1024 * 1024,
                                           backupCount=5)
        handlers.append(file_handler)
    logging.basicConfig(
        format='[{asctime}][{levelname}] - {name}: {message}',
        style='{',
        level=logging.getLevelName(level),
        handlers=handlers,
    )


class CNDProject:
    def __init__(self, name, video_path, save_path, fps=60, frame_size=(1280, 720), coord=(500, 500), answer=""):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)
        # self.ocr_stream = OcrStream(self.state, self.video_reader)
        self.answer=answer

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, fps, frame_size, coord, answer)
        self.logger.info("Start Project")

    def start(self):
        self.logger.info("Start project act start")
        try:
            self.video_reader.start()
            # self.ocr_stream.start()
            self.visualize_stream.start()
            self.state.exit_event.wait()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stop Project")

        self.video_reader.stop()
        # self.ocr_stream.stop()
        self.visualize_stream.stop()

    def get_best_accuracy(self):
        result = {
            'Best_Accuracy:': self.visualize_stream.best_accuracy / 6,
            'Best_Prediction': self.visualize_stream.best_result,
            'Correct_Result': self.answer,
            'Best_Frame_Num': self.visualize_stream.best_frame
        }
        return result


out_acc = {}
with open('config.json', 'r') as config_input:
    args = json.load(config_input)['data']


if __name__ == '__main__':
    setup_logging(sys.argv[1])
    logger = logging.getLogger(__name__)
    project = None
    for name, answer in args:
        logger.info('BEGIN ' + name)
        try:
            project = CNDProject("CNDProject", name, 'out' + name[1:], answer=answer)
            project.start()
        except Exception as e:
            logger.exception(e)
        finally:
            if project is not None:
                project.stop()
                out_acc[name[9:]] = project.get_best_accuracy()

result = json.dumps(out_acc, indent=4)
with open('./log/accuracy_results.txt', 'w') as f:
    f.write(result)
