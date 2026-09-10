"""No SDK/network connections. GUI subprocess regressions are opt-in."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from utils.core.process_viewer import ProcessViewer


def test_publish_drops_busy_frame_without_waiting_or_mutating_state():
    viewer = ProcessViewer.__new__(ProcessViewer)
    viewer._closed = False
    viewer._shared = np.array([1., 2., 3.])
    viewer._frame_lock = threading.Lock()
    with viewer._frame_lock:
        assert viewer.publish(np.array([4., 5., 6.])) is False
        np.testing.assert_array_equal(viewer._shared, [1., 2., 3.])
    pose = np.array([4., 5., 6.])
    assert viewer.publish(pose)
    pose[:] = 0  # Display slot owns its copy, not the control array.
    np.testing.assert_array_equal(viewer._shared, [4., 5., 6.])
    with pytest.raises(ValueError, match='Invalid viewer pose'):
        viewer.publish([1., float('nan'), 3.])
    with pytest.raises(ValueError, match='Invalid viewer pose'):
        viewer.publish([1.])
    viewer._closed = True
    assert viewer.publish([1., 2., 3.]) is False


def test_startup_failure_reaps_display_process(tmp_path):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match='viewer'):
        ProcessViewer(tmp_path / 'missing.xml', np.zeros(1), startup_timeout=10)
    assert {p.pid for p in mp.active_children()} <= before


@pytest.mark.skipif(os.environ.get('REALMAN_TEST_GUI') != '1',
                    reason='Set REALMAN_TEST_GUI=1 on a desktop for native GUI regression')
@pytest.mark.parametrize('fault', [False, True])
def test_native_viewer_normal_and_fault_exit_have_no_segfault(fault):
    # Use a fresh interpreter so a GLFW atexit crash fails this test, not pytest.
    script = r'''
import time
from example import teleop_realman_real as r
from utils.core.realman_input_stream import InputPollResult
from utils.core.safety import SafetyFault
class Input:
    def __init__(self, args): self.count = 0
    def poll_arms(self, left, right, _):
        self.count += 1
        if FAULT and self.count > 4:
            return InputPollResult(invalid_wrist_sides={'left'})
        for arm in (left, right):
            arm.update_from_pose((.3, .3, 1), (0, 0, 0, 1))
        now = time.monotonic()
        return InputPollResult(True, wrist_received_at={'left': now, 'right': now})
    def close(self): pass

def forbidden(**kw): raise AssertionError('No hardware allowed')
try:
    r.run(r.parse_args(['--dry-run', '--viewer', '--duration', '0.4']),
          input_factory=Input, output_factory=forbidden)
except SafetyFault as exc:
    if not FAULT or 'left: wrist' not in str(exc): raise
else:
    if FAULT: raise AssertionError('Fault was not propagated')
print('CLEAN_EXIT')
'''.replace('FAULT', repr(fault))
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CLEAN_EXIT' in result.stdout
    assert 'Segmentation fault' not in result.stderr
