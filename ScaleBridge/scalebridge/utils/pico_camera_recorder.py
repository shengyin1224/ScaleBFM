"""Internal camera worker started by PicoDatasetRecorder.

The default path consumes the same TWIST2 RealSense ZMQ multipart stream used
by SONIC.  A local OpenCV device remains available for explicit diagnostics.
"""
import pickle
import signal
import sys
import time

import cv2
import numpy as np


def _remote_frames(host: str, port: int):
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.connect(f"tcp://{host}:{port}")
    try:
        while True:
            parts = socket.recv_multipart()
            if len(parts) < 4 or parts[0] != b"realsense":
                continue
            try:
                pickle.loads(parts[1])
                frame = cv2.imdecode(np.frombuffer(parts[2], dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                    yield frame
            except Exception:
                continue
    finally:
        socket.close(linger=0)
        context.term()


def main() -> int:
    if len(sys.argv) < 4:
        return 2
    protocol = sys.argv[1]
    output, fps = sys.argv[2], float(sys.argv[3])
    # PicoDatasetRecorder stops this worker at episode end; translate SIGTERM
    # into KeyboardInterrupt so the writer's finally block is always run.
    def _stop_worker(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_worker)
    if protocol == "twist2_realsense":
        if len(sys.argv) < 6:
            return 2
        from gear_sonic.data.video_writer import VideoWriter

        frames = _remote_frames(sys.argv[4], int(sys.argv[5]))
        writer = VideoWriter(output, 640, 480, int(round(fps)), "h264")
        try:
            for frame in frames:
                writer.add_frame(frame)
        except KeyboardInterrupt:
            return 0
        finally:
            writer.stop()
        return 0

    source = int(protocol) if protocol.isdigit() else protocol
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        return 3
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        camera.release()
        return 4
    try:
        while True:
            ok, frame = camera.read()
            if ok:
                writer.write(frame)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        writer.release()
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
