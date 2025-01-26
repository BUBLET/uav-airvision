import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from src.image_processing.odometry_pipeline import OdometryPipeline
from src.image_processing.feature_extraction import FeatureExtractor
from src.image_processing.feature_matching import FeatureMatcher
from src.image_processing.odometry_calculation import OdometryCalculator
from src.image_processing.frame_processor import FrameProcessor
from src.image_processing.map_point import MapPoint
from src.image_processing.trajectory_writer import TrajectoryWriter


class TestOdometryPipeline(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = MagicMock(spec=FeatureExtractor)
        self.feature_matcher = MagicMock(spec=FeatureMatcher)
        self.odometry_calculator = MagicMock(spec=OdometryCalculator)
        self.frame_processor = MagicMock(spec=FrameProcessor)
        self.logger = MagicMock()
        self.metrics_logger = MagicMock()
        self.trajectory_writer = MagicMock(spec=TrajectoryWriter)
        self.pipeline = OdometryPipeline(
            feature_extractor=self.feature_extractor,
            feature_matcher=self.feature_matcher,
            odometry_calculator=self.odometry_calculator,
            frame_processor=self.frame_processor,
            logger=self.logger,
            metrics_logger=self.metrics_logger,
            lost_threshold=2,
            trajectory_writer=self.trajectory_writer
        )

    def test_initialize_success(self):
        self.feature_extractor.extract_features.return_value = ([MagicMock()], MagicMock())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.initialize(frame, frame_idx=1)
        self.assertEqual(len(self.pipeline.keyframes), 1)
        self.assertEqual(len(self.pipeline.poses), 1)
        self.trajectory_writer.write_pose.assert_called_once()

    def test_initialize_no_keypoints(self):
        self.feature_extractor.extract_features.return_value = ([], None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.pipeline.initialize(frame, frame_idx=1)

    def test_process_frame_when_uninitialized(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame_idx=2, current_frame=frame)
        self.logger.warning.assert_called()

    def test_process_frame_lost_frames_increment(self):
        self.pipeline.keyframes.append((1, [MagicMock()], MagicMock(), np.eye(3, 4)))
        self.pipeline.poses.append(np.eye(3, 4))
        self.frame_processor.process_frame.return_value = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame_idx=2, current_frame=frame)
        self.assertEqual(self.pipeline.lost_frames_count, 1)
        self.frame_processor.process_frame.return_value = None
        self.pipeline.process_frame(frame_idx=3, current_frame=frame)
        self.assertEqual(self.pipeline.lost_frames_count, 2)
        self.assertEqual(len(self.pipeline.keyframes), 1)

    def test_process_frame_success(self):
        self.pipeline.keyframes.append((1, [MagicMock()], MagicMock(), np.eye(3, 4)))
        self.pipeline.poses.append(np.eye(3, 4))
        self.frame_processor.process_frame.return_value = (
            [MagicMock()], MagicMock(), np.eye(3, 4), [MagicMock(spec=MapPoint)], True
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame_idx=2, current_frame=frame)
        self.assertEqual(self.pipeline.lost_frames_count, 0)
        self.assertEqual(len(self.pipeline.keyframes), 2)
        self.assertEqual(len(self.pipeline.poses), 2)
        self.trajectory_writer.write_pose.assert_called()

    def test_reset(self):
        self.pipeline.keyframes.append((1, [MagicMock()], MagicMock(), np.eye(3, 4)))
        self.pipeline.poses.append(np.eye(3, 4))
        self.pipeline.map_points.append(MagicMock())
        self.pipeline.initialization_completed = True
        self.pipeline._reset()
        self.assertFalse(self.pipeline.initialization_completed)
        self.assertEqual(len(self.pipeline.keyframes), 0)
        self.assertEqual(len(self.pipeline.poses), 0)
        self.assertEqual(len(self.pipeline.map_points), 0)
        self.assertEqual(self.pipeline.lost_frames_count, 0)

    @patch('cv2.VideoCapture')
    def test_run_video_ok(self, mock_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_capture.return_value = mock_cap_instance
        self.feature_extractor.extract_features.return_value = ([MagicMock()], MagicMock())
        self.frame_processor.process_frame.return_value = (
            [MagicMock()], MagicMock(), np.eye(3, 4), [MagicMock(spec=MapPoint)], True
        )
        self.pipeline.run('dummy_path')
        self.assertTrue(mock_cap_instance.isOpened.called)
        self.trajectory_writer.write_pose.assert_called()
        self.trajectory_writer.close.assert_called()

    @patch('cv2.VideoCapture')
    def test_run_no_video(self, mock_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_capture.return_value = mock_cap_instance
        self.pipeline.run('dummy_path')
        self.logger.error.assert_called()

    @patch('cv2.VideoCapture')
    def test_run_no_first_frame(self, mock_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (False, None)
        mock_capture.return_value = mock_cap_instance
        self.pipeline.run('dummy_path')
        self.logger.error.assert_called()

    @patch('cv2.VideoCapture')
    def test_run_end_of_video(self, mock_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_capture.return_value = mock_cap_instance
        self.feature_extractor.extract_features.return_value = ([MagicMock()], MagicMock())
        self.frame_processor.process_frame.return_value = (
            [MagicMock()], MagicMock(), np.eye(3, 4), [MagicMock(spec=MapPoint)], True
        )
        self.pipeline.run('dummy_path')
        self.logger.info.assert_any_call('Достигнут конец видео.')


if __name__ == '__main__':
    unittest.main()
