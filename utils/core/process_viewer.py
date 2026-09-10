"""Display-only MuJoCo viewer isolated from the robot control process.

Frames use a single shared-memory slot. Publishing never waits for rendering,
never queues old poses, and never copies GUI edits back to the controller.
The GUI process owns its model, data, GLFW context, and render threads.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import threading
import time
import traceback

import numpy as np

LOG = logging.getLogger(__name__)


def _viewer_main(scene_path, frames, frame_lock, stop, ready, status):
    # Lazy imports: the controller does not create a GLFW context or GUI thread.
    import mujoco
    from mujoco import viewer as mj_viewer

    # The parent handles Ctrl+C, stops hardware first, then closes this GUI.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    viewer = None
    owned_threads = set()
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        shared = np.frombuffer(frames, dtype=np.float64)
        if shared.shape != data.qpos.shape:
            raise ValueError("Viewer model joint count differs from controller")
        with frame_lock:
            data.qpos[:] = shared
        mujoco.mj_forward(model, data)
        before = set(threading.enumerate())
        viewer = mj_viewer.launch_passive(model, data)
        # MuJoCo 3.9 Handle.close() only requests exit. Join the threads created
        # by launch_passive before interpreter shutdown runs glfw.terminate().
        # They belong solely to this display process, not Pico or the SDK.
        owned_threads = set(threading.enumerate()) - before
        status.send(None)
        ready.set()
        while not stop.is_set() and viewer.is_running():
            with frame_lock:
                pose = shared.copy()
            data.qpos[:] = pose
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            stop.wait(1 / 30)
    except BaseException:
        try:
            status.send(traceback.format_exc())
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        ready.set()
        try:
            if viewer is not None:
                viewer.close()
        finally:
            for thread in owned_threads:
                thread.join()
            status.close()


class ProcessViewer:
    def __init__(self, scene_path, qpos, *, startup_timeout=30.0):
        initial = np.asarray(qpos, dtype=np.float64)
        if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
            raise ValueError("Expected a finite one-dimensional viewer pose")
        # Never fork a process holding active SDK/GL/thread state.
        ctx = mp.get_context("spawn")
        self._frames = ctx.RawArray("d", initial.size)
        self._shared = np.frombuffer(self._frames, dtype=np.float64)
        self._shared[:] = initial
        self._frame_lock = ctx.Lock()
        self._stop = ctx.Event()
        ready = ctx.Event()
        self._status, sender = ctx.Pipe(duplex=False)
        self._closed = False
        self._process = ctx.Process(
            name="RealMan-display-only", target=_viewer_main,
            args=(str(scene_path), self._frames, self._frame_lock,
                  self._stop, ready, sender), daemon=True,
        )
        try:
            self._process.start()
            sender.close()
            deadline = time.monotonic() + startup_timeout
            while not ready.wait(0.05):
                if not self._process.is_alive():
                    raise RuntimeError("MuJoCo viewer exited during startup")
                if time.monotonic() >= deadline:
                    raise RuntimeError("MuJoCo viewer startup timed out")
            if not self._status.poll():
                raise RuntimeError("MuJoCo viewer did not report successful startup")
            error = self._status.recv()
            if error:
                raise RuntimeError(f"MuJoCo viewer startup failed:\n{error}")
        except BaseException:
            self.close()
            raise
        finally:
            sender.close()

    def is_running(self):
        if self._closed:
            return False
        if self._status.poll():
            try:
                error = self._status.recv()
            except EOFError:
                error = None
            if error:
                raise RuntimeError(f"MuJoCo viewer failed:\n{error}")
        if self._process.is_alive():
            return True
        if self._process.exitcode != 0:
            raise RuntimeError(f"MuJoCo viewer exited with code {self._process.exitcode}")
        return False

    def publish(self, qpos):
        """Drop this display frame if busy; control must never wait on the GUI."""
        if self._closed:
            return False
        pose = np.asarray(qpos, dtype=np.float64)
        if pose.shape != self._shared.shape or not np.all(np.isfinite(pose)):
            raise ValueError("Invalid viewer pose")
        if not self._frame_lock.acquire(False):
            return False
        try:
            self._shared[:] = pose
        finally:
            self._frame_lock.release()
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                LOG.warning("Display did not close in time; terminating GUI process only")
                self._process.terminate()
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._status.close()
