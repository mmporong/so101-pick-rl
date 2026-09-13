"""Contact diagnostics must preserve signed distances and ignore unused buffers."""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab"))
from so101_pick_rl.grasp_audit import transform_points, unpack_contacts


class GraspAuditTests(unittest.TestCase):
    def test_identity_and_translation(self):
        np.testing.assert_allclose(transform_points([[1, 2, 3]], [1, -1, 2], [1, 0, 0, 0]), [[2, 1, 5]])

    def test_quaternion_is_wxyz(self):
        np.testing.assert_allclose(transform_points([[1, 0, 0]], [0, 0, 0], [2**-.5, 0, 0, 2**-.5]), [[0, 1, 0]], atol=1e-12)

    def test_nonunit_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            transform_points([[0, 0, 0]], [0, 0, 0], [2, 0, 0, 0])

    def buffers(self):
        return [np.array([[99.], [2.], [3.], [np.nan]]), np.zeros((4, 3)), np.zeros((4, 3)),
                np.array([[-99.], [-.002], [.003], [np.nan]]), np.array([[2, 0]]), np.array([[1, 0]])]

    def test_active_signed_separation_only(self):
        result = unpack_contacts(self.buffers())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["separation_m"], [-.002, .003])
        self.assertEqual(result[0]["normal_force_n"], [2., 3.])

    def test_empty_contacts_are_not_zero_penetration(self):
        values = self.buffers()
        values[4][:] = 0
        self.assertEqual(unpack_contacts(values), [])

    def test_overflow_rejected(self):
        values = self.buffers()
        values[4][0, 0] = 4
        with self.assertRaises(ValueError):
            unpack_contacts(values)

    def test_nonfinite_active_value_rejected(self):
        values = self.buffers()
        values[0][1] = np.nan
        with self.assertRaises(ValueError):
            unpack_contacts(values)


if __name__ == "__main__":
    unittest.main()
