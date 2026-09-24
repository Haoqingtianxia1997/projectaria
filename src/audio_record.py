import argparse
import os
import sys
import threading
from math import gcd
from pathlib import Path
from typing import Optional

import time
import sounddevice as sd

import aria.sdk as aria
import numpy as np
import rclpy
import whisper
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt
from std_msgs.msg import Bool, Empty, String

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord
from src.gaze_rgb_config import TRANSCRIPTION_HOLD_SEC
from utils.aria_rgb_stream import AriaStream

ARIA_AUDIO_SAMPLE_RATE = 48000
ARIA_NUM_CHANNELS = 7
WHISPER_SAMPLE_RATE = 16000
AUDIO_GAP_TOLERANCE_SAMPLES = 2

# /transcription must be published after /gaze_label (the reader on the other device
# uses the transcription to decide whether to consume the gaze label). The gaze
# pipeline publishes /gaze_label after stitching + peak analysis on B release; this is
# the longest we hold the transcription waiting for it before giving up.
GAZE_LABEL_TIMEOUT_SEC = 30.0

# A transcription containing this word is a stop command, not an instruction:
# it goes out on /reset_switch instead of /transcription.
STOP_KEYWORD = "stop"

# Local headset capture (via PulseAudio). The raw ALSA device is held exclusively
# by PulseAudio, so we record through the "default" pulse source: plugging in the
# H390 makes it the system default input, so no hardware name needs hard-coding.
# Override with --pulse-source if a non-default source is ever needed.
LOCAL_SAMPLE_RATE = 44100

def beep(freq: float = 440.0, duration: float = 0.1, volume: float = 0.5, sample_rate: int = 44100) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()


def beep_n(times: int = 1, freq: float = 880.0, duration: float = 0.2, interval: float = 0.1) -> None:
    for i in range(times):
        beep(freq=freq, duration=duration)
        if i < times - 1:
            time.sleep(interval)

def play_calib_beep():
    beep_n(times=1, duration=0.2)
    time.sleep(0.1)


class AudioHandler:
    """Receives audio frames from the shared AriaStream, drives the record / transcribe
    state machine over ROS, and writes wav + publishes /transcription on B release.

    Does NOT own a StreamingClient. on_audio_received(...) runs on the SDK's audio
    thread; the recording flag and audio buffer are touched from that thread and
    from ROS callbacks running in this handler's own background spin thread.
    """

    def __init__(
        self,
        channels: Optional[list[int]] = None,
        gain: float = 2.0,
        language: str = "en",
        participant: str = "",
        lowcut: float = 100.0,
        highcut: float = 4000.0,
        filter_order: int = 6,
        whisper_model: str = "small",
        source: str = "aria",
    ) -> None:
        self.channels = channels
        self.gain = gain
        self.language = language

        # Input sample rate depends on the capture source: Aria glasses (48 kHz,
        # 7 ch int32) vs. the local H390 headset (44.1 kHz, mono float32).
        self.source = source
        self.input_sample_rate = LOCAL_SAMPLE_RATE if source == "local" else ARIA_AUDIO_SAMPLE_RATE

        self.audio_buffer: list = []
        self._buffer_lock = threading.Lock()
        self.recording: bool = False

        # Local-capture InputStream (created lazily in start_local_capture()).
        self._local_stream: Optional[sd.InputStream] = None

        print("Loading whisper model...")
        self.model = whisper.load_model(whisper_model)

        self._g = gcd(self.input_sample_rate, WHISPER_SAMPLE_RATE)
        self._sos = butter(
            filter_order, [lowcut, highcut], btype="band",
            fs=self.input_sample_rate, output="sos",
        )
        print(f"Bandpass filter: {lowcut}–{highcut} Hz, order {filter_order} "
              f"(source={source}, {self.input_sample_rate} Hz)")

        self._participant_base: Optional[Path] = None
        if participant:
            self._participant_base = Path("recordings") / participant
            self._participant_base.mkdir(parents=True, exist_ok=True)
            print(f"Session folder: {self._participant_base}")
        self._current_session_dir: Optional[Path] = None

        self._init_ros()

    # ------------------------------------------------------------------
    # ROS plumbing
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("audio_transcriber")
        self._pub_text = self._node.create_publisher(String, "/transcription", 10)
        self._pub_stop = self._node.create_publisher(Bool, "/reset_switch", 10)
        self._pub_start = self._node.create_publisher(Empty, "/recording/start", 10)
        # Published once the calibration beep ends, so gaze_label accumulation begins
        # after the beep rather than at video-recording start.
        self._pub_gaze_label_start = self._node.create_publisher(
            Empty, "/gaze_label_recording_start", 10
        )
        self._node.create_subscription(Empty, "/key/b/press", self._on_b_press, 10)
        self._node.create_subscription(Empty, "/key/b/release", self._on_b_release, 10)
        # Set once the gaze pipeline publishes its post-release /gaze_label result;
        # _do_transcribe waits on it so /transcription always goes out afterwards.
        self._gaze_label_event = threading.Event()
        self._node.create_subscription(String, "/gaze_label", self._on_gaze_label, 10)
        # Dedicated executor so we don't fight other modules over the rclpy
        # global executor when running under main_entry.py.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._ros_thread = threading.Thread(
            target=self._executor.spin, daemon=True,
        )
        self._ros_thread.start()
        print("Press [B] to record, release to transcribe. Ctrl+C to quit.")

    def _on_b_press(self, _msg: Empty) -> None:
        if self._participant_base is not None:
            existing = [
                int(p.name) for p in self._participant_base.iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
            idx = max(existing, default=0) + 1
            session_dir = self._participant_base / f"{idx:02d}"
            session_dir.mkdir()
            self._current_session_dir = session_dir
        self._pub_start.publish(Empty())
        with self._buffer_lock:
            self.audio_buffer.clear()
            play_calib_beep()
            self.recording = True
        # Beep finished — signal gaze_label accumulation to start now.
        self._pub_gaze_label_start.publish(Empty())
        print("Recording...")

    def _on_b_release(self, _msg: Empty) -> None:
        self.recording = False
        # Arm the wait: only a /gaze_label published after this release counts
        # (the gaze pipeline needs at least stitch+peak-analysis time, so nothing from
        # this session can arrive before the clear).
        self._gaze_label_event.clear()
        threading.Thread(target=self._do_transcribe, daemon=True).start()

    def _on_gaze_label(self, msg: String) -> None:
        if not self._gaze_label_event.is_set():
            print(f"/gaze_label received ({len(msg.data)} bytes); /transcription unblocked.")
        self._gaze_label_event.set()

    # ------------------------------------------------------------------
    # SDK audio callback
    # ------------------------------------------------------------------

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord) -> None:
        if not self.recording:
            return
        try:
            samples = np.array(audio_data.data, dtype=np.int32)
            timestamps = np.array(record.capture_timestamps_ns, dtype=np.int64)
            with self._buffer_lock:
                if self.recording:
                    self.audio_buffer.append((samples, timestamps))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Local H390 capture (sounddevice / PulseAudio)
    # ------------------------------------------------------------------

    def _on_local_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback: buffer mono float32 frames while recording."""
        if not self.recording:
            return
        # indata is (frames, 1) float32; flatten to mono and copy (the buffer is reused).
        chunk = indata[:, 0].copy()
        with self._buffer_lock:
            if self.recording:
                self.audio_buffer.append(chunk)

    def start_local_capture(self) -> None:
        """Open the local input stream. Uses PulseAudio's default source (the H390
        once plugged in), unless PULSE_SOURCE was set in main() to override it."""
        self._local_stream = sd.InputStream(
            device="default",
            channels=1,
            samplerate=self.input_sample_rate,
            dtype="float32",
            callback=self._on_local_audio,
        )
        self._local_stream.start()
        override = os.environ.get("PULSE_SOURCE")
        print(f"Local capture started (source={override or 'PulseAudio default'}).")

    def stop_local_capture(self) -> None:
        if self._local_stream is not None:
            try:
                self._local_stream.stop()
                self._local_stream.close()
            except Exception:
                pass
            self._local_stream = None

    # ------------------------------------------------------------------
    # Transcription pipeline
    # ------------------------------------------------------------------

    def _do_transcribe(self) -> None:
        print("Stopped.")

        with self._buffer_lock:
            chunks = list(self.audio_buffer)

        if not chunks:
            print("No audio data received.")
            print("Press [B] to record again. Ctrl+C to quit.")
            return

        if self.source == "local":
            audio_float = self._local_audio_float(chunks)
        else:
            audio_float = self._aria_audio_float(chunks)
        if audio_float is None or audio_float.size == 0:
            print("No complete audio frames received.")
            print("Press [B] to record again. Ctrl+C to quit.")
            return

        audio_float = sosfilt(self._sos, audio_float).astype(np.float32)

        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 1e-6:
            audio_float = audio_float / rms * 0.1
        audio_float = np.clip(audio_float * self.gain, -1.0, 1.0)

        audio_16k = resample_poly(
            audio_float,
            WHISPER_SAMPLE_RATE // self._g,
            self.input_sample_rate // self._g,
        ).astype(np.float32)

        if self._current_session_dir is not None:
            wav_path = self._current_session_dir / "audio.wav"
            wavfile.write(wav_path, WHISPER_SAMPLE_RATE, audio_16k)
            print(f"Saved: {wav_path}")

        print("Transcribing...")
        result = self.model.transcribe(
            audio_16k, 
            fp16=False, 
            language=self.language, 
            temperature=0.2, 
            beam_size=10,
            condition_on_previous_text=False
            )
        text = result["text"].strip()

        # A stop command is not an instruction for the other device to act on:
        # publish /reset_switch instead of /transcription, and skip the /gaze_label
        # wait entirely (nothing downstream is ordering against a label here).
        if STOP_KEYWORD in text.lower():
            stop_msg = Bool()
            stop_msg.data = True
            self._pub_stop.publish(stop_msg)
            print(f"Text: {text}")
            print("Stop keyword detected; published /reset_switch (no /transcription).")
            print("Press [B] to record again. Ctrl+C to quit.")
            return

        # Hold /transcription until the gaze pipeline has published /gaze_label
        # (it always publishes after B release, empty data on failure), so the
        # reader on the other device sees the label result before the text.
        wait_start = time.monotonic()
        if not self._gaze_label_event.wait(timeout=GAZE_LABEL_TIMEOUT_SEC):
            print("Timed out waiting for /gaze_label; publishing /transcription anyway.")
        else:
            waited = time.monotonic() - wait_start
            if waited > 0.05:
                print(f"Held /transcription {waited:.2f}s for /gaze_label.")
            # The event fires when THIS node sees /gaze_label; the device reading
            # /transcription subscribes independently and may lag by a moment.
            # A short grace (well inside the GAZE_LABEL_TAIL_SEC re-publish tail)
            # keeps the ordering true on that device too.
            time.sleep(TRANSCRIPTION_HOLD_SEC)

        msg = String()
        msg.data = text
        self._pub_text.publish(msg)

        print(f"Text: {text}")
        print("Press [B] to record again. Ctrl+C to quit.")

    def _aria_audio_float(self, chunks: list) -> Optional[np.ndarray]:
        """Aria path: timestamp-align 7-channel int32 frames, mix selected channels,
        and convert to mono float32 at ARIA_AUDIO_SAMPLE_RATE (pre-filter)."""
        frames, inserted_samples, dropped_overlap = self._timestamp_aligned_frames(chunks)
        if frames.size == 0:
            return None
        if inserted_samples:
            inserted_ms = inserted_samples / ARIA_AUDIO_SAMPLE_RATE * 1000.0
            print(
                f"Audio timestamp alignment inserted {inserted_samples} "
                f"samples ({inserted_ms:.1f} ms) of silence."
            )
        if dropped_overlap:
            dropped_ms = dropped_overlap / ARIA_AUDIO_SAMPLE_RATE * 1000.0
            print(
                f"Audio timestamp alignment dropped {dropped_overlap} "
                f"overlapping samples ({dropped_ms:.1f} ms)."
            )

        channels = self.channels if self.channels is not None else list(range(ARIA_NUM_CHANNELS))
        channel_data = frames[:, channels].mean(axis=1).astype(np.int32)
        return (channel_data >> 16).astype(np.float32) / 32768.0

    def _local_audio_float(self, chunks: list) -> Optional[np.ndarray]:
        """Local path: concatenate the buffered mono float32 H390 chunks."""
        return np.concatenate(chunks).astype(np.float32)

    def _timestamp_aligned_frames(
        self,
        chunks: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, int, int]:
        aligned: list[np.ndarray] = []
        prev_last_ts_ns: Optional[float] = None
        inserted_samples = 0
        dropped_overlap = 0
        sample_period_ns = 1e9 / ARIA_AUDIO_SAMPLE_RATE

        for samples, timestamps in chunks:
            frames = self._reshape_interleaved_audio(samples)
            if frames.size == 0:
                continue

            start_ts_ns = self._chunk_start_timestamp_ns(timestamps, len(frames))
            if prev_last_ts_ns is not None and start_ts_ns is not None:
                elapsed_samples = int(
                    round((start_ts_ns - prev_last_ts_ns) / sample_period_ns)
                )
                missing_samples = elapsed_samples - 1
                if missing_samples > AUDIO_GAP_TOLERANCE_SAMPLES:
                    aligned.append(
                        np.zeros((missing_samples, ARIA_NUM_CHANNELS), dtype=np.int32)
                    )
                    inserted_samples += missing_samples
                elif missing_samples < -AUDIO_GAP_TOLERANCE_SAMPLES:
                    overlap = min(-missing_samples, len(frames))
                    frames = frames[overlap:]
                    dropped_overlap += overlap
                    start_ts_ns += overlap * sample_period_ns
                    if frames.size == 0:
                        continue

            aligned.append(frames)
            if start_ts_ns is not None:
                prev_last_ts_ns = start_ts_ns + (len(frames) - 1) * sample_period_ns

        if not aligned:
            return np.empty((0, ARIA_NUM_CHANNELS), dtype=np.int32), 0, 0
        return np.concatenate(aligned, axis=0), inserted_samples, dropped_overlap

    def _reshape_interleaved_audio(self, samples: np.ndarray) -> np.ndarray:
        remainder = len(samples) % ARIA_NUM_CHANNELS
        if remainder != 0:
            samples = samples[:-remainder]
        if len(samples) == 0:
            return np.empty((0, ARIA_NUM_CHANNELS), dtype=np.int32)
        return samples.reshape(-1, ARIA_NUM_CHANNELS)

    def _chunk_start_timestamp_ns(
        self,
        timestamps: np.ndarray,
        frame_count: int,
    ) -> Optional[float]:
        if timestamps.size == 0 or frame_count == 0:
            return None
        return float(timestamps[0])

    def shutdown(self) -> None:
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from Aria glass microphones and transcribe")
    parser.add_argument("--update_iptables", default=False, action="store_true",
                        help="Update iptables to enable receiving the data stream, only for Linux.")
    parser.add_argument("--device-ip", default="192.168.8.117",
                        help="IP address to connect to the device over wifi")
    parser.add_argument("--channel", type=int, nargs="+", default=None,
                        choices=range(ARIA_NUM_CHANNELS), metavar="{0..6}",
                        help=f"Microphone channel(s) to mix (0-{ARIA_NUM_CHANNELS - 1}); multiple allowed; default: all 7")
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Extra gain multiplier applied after normalization (default: 2.0)")
    parser.add_argument("--language", default="en",
                        help="Language for whisper transcription (default: en)")
    parser.add_argument("--participant", default="",
                        help="Participant ID (e.g. AB12). When set, creates the session subfolder on B press.")
    parser.add_argument("--lowcut", type=float, default=300.0,
                        help="Bandpass lower cutoff frequency in Hz (default: 100)")
    parser.add_argument("--highcut", type=float, default=5000.0,
                        help="Bandpass upper cutoff frequency in Hz (default: 4000)")
    parser.add_argument("--filter-order", type=int, default=6,
                        help="Butterworth filter order (default: 6)")
    parser.add_argument("--source", choices=["aria", "local"], default="aria",
                        help="Audio source: 'aria' (glasses, 48kHz 7ch) or "
                             "'local' (headset via PulseAudio default source, 44.1kHz mono). Default: aria")
    parser.add_argument("--pulse-source", default=None,
                        help="Override the PulseAudio capture source name for --source local "
                             "(default: system default source, i.e. the plugged-in headset).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Only override PulseAudio's default capture source if explicitly requested;
    # otherwise the system default (the plugged-in headset) is used. libpulse reads
    # PULSE_SOURCE when the stream connects.
    if args.source == "local" and args.pulse_source:
        os.environ["PULSE_SOURCE"] = args.pulse_source

    handler = AudioHandler(
        channels=args.channel,
        gain=args.gain,
        language=args.language,
        participant=args.participant,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
        source=args.source,
    )

    if args.source == "local":
        # No Aria stream needed; capture locally and keep the process alive while
        # the ROS spin thread drives the B-key record/transcribe state machine.
        handler.start_local_capture()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            handler.stop_local_capture()
            handler.shutdown()
        return

    stream = AriaStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        data_types=aria.StreamingDataType.Audio,
    )
    stream.add_audio_handler(handler)
    try:
        stream.run()
    finally:
        handler.shutdown()


if __name__ == "__main__":
    main()
