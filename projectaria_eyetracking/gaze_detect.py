import argparse
import sys
import threading
import time
from collections import deque

import aria.sdk as aria
import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from utils.common import update_iptables
from utils.aria_rgb_stream import crop_fisheye_img
from projectaria_eyetracking.inference.infer import EyeGazeInference
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)


CALIB_MARKER_ID: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=(
            "projectaria_eyetracking/inference/model/"
            "pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
        ),
        help="Location of the model weights",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=(
            "projectaria_eyetracking/inference/model/"
            "pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
        ),
        help="Location of the model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (cpu or cuda:0)",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--smooth-window",
        type=float,
        default=0.5,
        help="Sliding-window duration (seconds) to average pitch/yaw before publishing.",
    )
    parser.add_argument(
        "--gaze-buffer-delay-frames",
        type=int,
        default=0,
        help=(
            "Publish the gaze result this many EyeTrack results behind the newest "
            "result (0 publishes immediately)."
        ),
    )
    parser.add_argument(
        "--undistort-width",
        type=int,
        default=1408,
        help="Width of undistorted RGB image (must match gaze_rgb_visualizer).",
    )
    parser.add_argument(
        "--undistort-height",
        type=int,
        default=1408,
        help="Height of undistorted RGB image (must match gaze_rgb_visualizer).",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="test_homography/homography.txt",
        help="Path to homography matrix file (e.g. test_homography/homography.txt).",
    )
    parser.add_argument(
        "--device-ip",
        type=str,
        default=None,
        help="IP address of the Aria device (e.g. 192.168.8.117).",
    )
    parser.add_argument(
        "--calib-duration",
        type=float,
        default=40.0,
        help="Seconds to collect calibration samples after pressing C (default: 30).",
    )
    return parser.parse_args()


def _load_camera_matrix(undistort_width: int, undistort_height: int, device_ip: str | None = None):
    """Load RGB calibration from device and return (rgb_calib, dst_calib, camera_matrix).
    Falls back to a reasonable default if the device connection fails."""
    rgb_calib = None

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if device_ip:
        client_config.ip_v4_address = device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()
    sensors_calib_json = device.streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    device_client.disconnect(device)
    src_w, _ = rgb_calib.get_image_size()
    src_focal = rgb_calib.get_focal_lengths()[0]
    focal_length = src_focal * undistort_width / src_w
    print(f"RGB calib loaded: focal_length={focal_length:.2f} px")


    dst_calib = get_linear_camera_calibration(
        undistort_width, undistort_height, focal_length, "camera-rgb"
    )
    fx, fy = dst_calib.get_focal_lengths()
    cx, cy = dst_calib.get_principal_point()
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    print(f"camera_matrix: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    return rgb_calib, dst_calib, camera_matrix


def _setup_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
    else:
        detector = None
    return aruco, dictionary, params, detector


def main() -> None:
    args = parse_args()
    if args.gaze_buffer_delay_frames < 0:
        raise ValueError("--gaze-buffer-delay-frames must be >= 0")
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    try:
        import rclpy
        from geometry_msgs.msg import Vector3
        from std_msgs.msg import Empty

        rclpy.init(args=None)
        ros_node = rclpy.create_node("eye_gaze_publisher")
        gaze_topic = "/aria/gaze_euler"
        gaze_publisher = ros_node.create_publisher(Vector3, gaze_topic, 10)
        print(f"ROS2 publishing enabled: {gaze_topic}")
    except Exception as exc:
        raise RuntimeError(f"ROS2 publisher unavailable: {exc}") from exc

    inference_model = EyeGazeInference(
        args.model_checkpoint_path, args.model_config_path, args.device
    )

    # Load RGB calibration for ArUco-based calibration
    rgb_calib, dst_calib, camera_matrix = _load_camera_matrix(
        args.undistort_width, args.undistort_height, args.device_ip
    )
    aruco, aruco_dict, aruco_params, aruco_detector = _setup_aruco_detector()

    # Optional homography for RGB camera shift correction
    H_warp = None
    if args.homography is not None:
        H_warp = np.loadtxt(args.homography)
        print(f"Loaded homography from {args.homography}")

    streaming_client = aria.StreamingClient()

    def _apply_subscription() -> None:
        config = streaming_client.subscription_config
        config.subscriber_data_type = (
            aria.StreamingDataType.EyeTrack | aria.StreamingDataType.Rgb
        )
        config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1

        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        streaming_client.subscription_config = config

    # Keep both camera streams active so their device capture timestamps remain
    # available on the same Aria hardware clock throughout the run.
    _apply_subscription()

    class StreamingClientObserver:
        def __init__(self):
            self.images = {}
            self.rgb_sample = None

        def on_image_received(self, image: np.ndarray, record) -> None:
            sample = (int(record.capture_timestamp_ns), image)
            if record.camera_id == aria.CameraId.EyeTrack:
                self.images[record.camera_id] = sample
            elif record.camera_id == aria.CameraId.Rgb:
                self.rgb_sample = sample

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to image data")
    streaming_client.subscribe()

    # eyetrack_window = "Aria EyeTrack"

    # cv2.namedWindow(eyetrack_window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(eyetrack_window, 640, 240)
    # cv2.setWindowProperty(eyetrack_window, cv2.WND_PROP_TOPMOST, 1)
    # cv2.moveWindow(eyetrack_window, 50, 800)

    # Sliding-window state: keep (timestamp, pitch_cal, yaw_cal) for the past
    # `smooth_window` seconds and publish their average.
    smooth_window: float = args.smooth_window
    gaze_history: deque[tuple[float, float, float]] = deque()
    # Each entry is (EyeTrack capture timestamp, smoothed pitch, smoothed yaw).
    # Keeping N entries queued makes publication lag the newest result by N
    # EyeTrack inference results.
    gaze_publish_buffer: deque[tuple[int, float, float]] = deque()

    # Calibration state
    calibrating = False
    compute_calib = False
    calib_samples: list[tuple[float, float, float, float]] = []  # (marker_col, marker_row, pitch_raw, yaw_raw)
    pitch_offset = 0.0
    yaw_offset = 0.0
    _crop_ox = 0
    _crop_oy = 0
    _calib_lock = threading.Lock()
    _calib_timer: threading.Timer | None = None
    calib_duration: float = args.calib_duration

    def _stop_calib() -> None:
        nonlocal calibrating, compute_calib, _calib_timer
        with _calib_lock:
            if not calibrating:
                return
            calibrating = False
            compute_calib = True
            _calib_timer = None
        print(f"Calibration auto-stopped after {calib_duration:.0f}s.")

    def _on_calib_start(_msg) -> None:
        nonlocal calibrating, compute_calib, _calib_timer
        with _calib_lock:
            if calibrating:
                # second press within calibration window — stop early
                if _calib_timer is not None:
                    _calib_timer.cancel()
                    _calib_timer = None
                calibrating = False
                compute_calib = True
                print("Calibration manually stopped.")
                return
            calib_samples.clear()
            calibrating = True
        print(
            f"Calibrating: look at marker {CALIB_MARKER_ID} center and move your head; "
            f"auto-stops in {calib_duration:.0f}s, or press [ again to stop early..."
        )
        t = threading.Timer(calib_duration, _stop_calib)
        t.daemon = True
        t.start()
        _calib_timer = t

    ros_node.create_subscription(Empty, "/key/leftbrace", _on_calib_start, 10)
    threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

    # Latest raw gaze (updated each EyeTrack frame, used when pairing with RGB)
    latest_pitch_raw: float = 0.0
    latest_yaw_raw: float = 0.0
    latest_rgb_timestamp_ns: int | None = None

    print(
        f"Press C to start calibration (look at ArUco marker {CALIB_MARKER_ID}); "
        f"auto-stops after {calib_duration:.0f}s. 'q'/ESC to quit."
    )

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            if compute_calib:
                with _calib_lock:
                    compute_calib = False
                    samples_snapshot = list(calib_samples)
                if len(samples_snapshot) >= 2:
                    rows = np.array([s[1] for s in samples_snapshot])
                    cols = np.array([s[0] for s in samples_snapshot])
                    pitches = np.array([s[2] for s in samples_snapshot])
                    yaws = np.array([s[3] for s in samples_snapshot])
                    fx = camera_matrix[0, 0]
                    fy = camera_matrix[1, 1]
                    cx = camera_matrix[0, 2] - _crop_ox
                    cy = camera_matrix[1, 2] - _crop_oy

                    res_p = minimize_scalar(
                        lambda dp: float(np.sum((rows - cy + fy * np.tan(pitches - dp)) ** 2)),
                        bounds=(-np.pi, np.pi),
                        method="bounded",
                    )
                    res_y = minimize_scalar(
                        lambda dy: float(
                            np.sum(
                                (cols - cx - fx * np.tan(yaws - dy)) ** 2
                            )
                        ),
                        bounds=(-np.pi, np.pi),
                        method="bounded",
                    )
                    pitch_offset = float(res_p.x)
                    yaw_offset = float(res_y.x)
                    print(
                        f"Calibration done ({len(samples_snapshot)} samples). "
                        f"pitch_offset={pitch_offset:.4f}, yaw_offset={yaw_offset:.4f}"
                    )
                else:
                    print("Not enough samples for calibration (need >= 2).")

            # Consume RGB continuously. The image itself is processed only for
            # calibration, but every frame retains its hardware timestamp.
            rgb_sample = observer.rgb_sample
            observer.rgb_sample = None
            if rgb_sample is not None:
                rgb_timestamp_ns, bgr = rgb_sample
                latest_rgb_timestamp_ns = rgb_timestamp_ns
                # print(f"RGB capture_timestamp_ns={rgb_timestamp_ns}")

            if calibrating and rgb_sample is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb_calib is not None:
                    rgb = distort_by_calibration(rgb, dst_calib, rgb_calib)
                    
                # Rotate to match gaze projection display coordinate system
                display = np.ascontiguousarray(np.rot90(rgb, -1))
                
                display, _crop_ox, _crop_oy = crop_fisheye_img(display)
                
                # Apply homography warp if available
                if H_warp is not None:
                    h_d, w_d = display.shape[:2]
                    display = cv2.warpPerspective(display, H_warp, (w_d, h_d))
                    
                gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
                if aruco_detector is not None:
                    corners, ids, _ = aruco_detector.detectMarkers(gray)
                else:
                    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
                if ids is not None:
                    for mid, corner_set in zip(ids.flatten(), corners):
                        if int(mid) == CALIB_MARKER_ID:
                            pts = corner_set.reshape(-1, 2)
                            center_col = float(np.mean(pts[:, 0]))
                            center_row = float(np.mean(pts[:, 1]))
                            calib_samples.append(
                                (center_col, center_row, latest_pitch_raw, latest_yaw_raw)
                            )
                            print(
                                f"  sample {len(calib_samples)}: "
                                f"marker=({center_col:.0f},{center_row:.0f}) "
                                f"pitch_raw={latest_pitch_raw:.4f} yaw_raw={latest_yaw_raw:.4f}"
                            )
                            break

            if aria.CameraId.EyeTrack in observer.images:
                eye_timestamp_ns, eyetrack_image = observer.images[aria.CameraId.EyeTrack]
                del observer.images[aria.CameraId.EyeTrack]

                timestamp_text = (
                    str(latest_rgb_timestamp_ns)
                    if latest_rgb_timestamp_ns is not None
                    else "unavailable"
                )
                # print(
                #     f"EyeTrack capture_timestamp_ns={eye_timestamp_ns}, "
                #     f"latest RGB capture_timestamp_ns={timestamp_text}"
                # )

                # cv2.imshow(eyetrack_window, eyetrack_image)

                # The model expects a single grayscale image containing [left | right] eyes.
                eye_tensor = torch.from_numpy(eyetrack_image)

                preds, lower, upper = inference_model.predict(eye_tensor)
                yaw_raw = - preds[0][0].item()
                pitch_raw = preds[0][1].item()
                latest_pitch_raw = pitch_raw
                latest_yaw_raw = yaw_raw

                pitch_cal = pitch_raw - pitch_offset
                yaw_cal = yaw_raw - yaw_offset

                # Sliding window: average over the past `smooth_window` seconds.
                now = time.monotonic()
                gaze_history.append((now, pitch_cal, yaw_cal))
                cutoff = now - smooth_window
                while gaze_history and gaze_history[0][0] < cutoff:
                    gaze_history.popleft()
                pitch = sum(s[1] for s in gaze_history) / len(gaze_history)
                yaw = sum(s[2] for s in gaze_history) / len(gaze_history)

                gaze_publish_buffer.append((eye_timestamp_ns, pitch, yaw))
                delay_frames = args.gaze_buffer_delay_frames
                if len(gaze_publish_buffer) <= delay_frames:
                    print(
                        "Gaze publish buffer warming up: "
                        f"{len(gaze_publish_buffer)}/{delay_frames} frame(s)"
                    )
                else:
                    publish_timestamp_ns, publish_pitch, publish_yaw = (
                        gaze_publish_buffer.popleft()
                    )
                    buffered_duration_ms = (
                        eye_timestamp_ns - publish_timestamp_ns
                    ) / 1e6
                    print(
                        f"Publishing pitch={publish_pitch:.4f}, "
                        f"yaw={publish_yaw:.4f}, "
                        # f"EyeTrack capture_timestamp_ns={publish_timestamp_ns}, "
                        # f"delay_frames={delay_frames}, "
                        # f"buffered_device_time={buffered_duration_ms:.2f} ms"
                        + (" [calibrating]" if calibrating else "")
                    )

                    gaze_msg = Vector3()
                    gaze_msg.x = publish_pitch
                    gaze_msg.y = publish_yaw
                    gaze_msg.z = 0.0
                    gaze_publisher.publish(gaze_msg)

            time.sleep(0.001)
    finally:
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        try:
            ros_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
