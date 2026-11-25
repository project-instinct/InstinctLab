import numpy as np

from instinctlab.motion_reference.motion_files.contact_motion import next_onset_index


def test_next_onset_index():
    flag = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0]], dtype=bool)
    # body 0: onsets at frames 0 and 4; body 1: onset at frame 2; T = 6 where there is none
    expected = np.array([[0, 2], [4, 2], [4, 2], [4, 6], [4, 6], [6, 6]])
    np.testing.assert_array_equal(next_onset_index(flag), expected)
