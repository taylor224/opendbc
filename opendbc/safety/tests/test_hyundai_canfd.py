#!/usr/bin/env python3
from opendbc.testing import parameterized_class, parameterized
import unittest
import numpy as np

from opendbc.car.hyundai.carcontroller import ANGLE_SAFETY_BASELINE_MODEL
from opendbc.car.hyundai.values import HyundaiSafetyFlags, CAR, HyundaiFlags, CarControllerParams
from opendbc.sunnypilot.car.hyundai.values import ANGLE_STEERING_MODEL_BY_CAR, encode_angle_model_id
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel, calc_slip_factor
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import away_round, round_speed
from opendbc.safety.tests.hyundai_common import HyundaiButtonBase, HyundaiLongitudinalBase, Buttons, PREV_BUTTON_SAMPLES
from opendbc.sunnypilot.car.hyundai.values import HyundaiSafetyFlagsSP
from opendbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm, AngleSteeringLimitsVM
from opendbc.car.hyundai.interface import CarInterface

# All combinations of radar/camera-SCC and gas/hybrid/EV cars
ALL_GAS_EV_HYBRID_COMBOS = [
  # Radar SCC
  {"GAS_MSG": ("ACCELERATOR_BRAKE_ALT", "ACCELERATOR_PEDAL_PRESSED"), "SCC_BUS": 0, "EXTRA_SAFETY_PARAM": 0},
  {"GAS_MSG": ("ACCELERATOR", "ACCELERATOR_PEDAL"), "SCC_BUS": 0, "EXTRA_SAFETY_PARAM": HyundaiSafetyFlags.EV_GAS},
  {"GAS_MSG": ("ACCELERATOR_ALT", "ACCELERATOR_PEDAL"), "SCC_BUS": 0, "EXTRA_SAFETY_PARAM": HyundaiSafetyFlags.HYBRID_GAS},
  # Camera SCC
  {"GAS_MSG": ("ACCELERATOR_BRAKE_ALT", "ACCELERATOR_PEDAL_PRESSED"), "SCC_BUS": 2, "EXTRA_SAFETY_PARAM": HyundaiSafetyFlags.CAMERA_SCC},
  {"GAS_MSG": ("ACCELERATOR", "ACCELERATOR_PEDAL"), "SCC_BUS": 2, "EXTRA_SAFETY_PARAM": HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CAMERA_SCC},
  {"GAS_MSG": ("ACCELERATOR_ALT", "ACCELERATOR_PEDAL"), "SCC_BUS": 2, "EXTRA_SAFETY_PARAM": HyundaiSafetyFlags.HYBRID_GAS | HyundaiSafetyFlags.CAMERA_SCC},
]

CAMERA_SCC_COMBOS = ALL_GAS_EV_HYBRID_COMBOS[3:]


def round_angle(angle_deg: float, can_offset=0):
  scaled = angle_deg / 0.1
  scaled += can_offset
  return int(scaled) * 0.1


class TestHyundaiCanfdBase(HyundaiButtonBase, common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):

  DBC = "hyundai_canfd_generated"
  SAFETY_MODEL = CarParams.SafetyModel.hyundaiCanfd

  TX_MSGS = [[0x50, 0], [0x1CF, 1], [0x2A4, 0]]
  STANDSTILL_THRESHOLD = 0.375 * 0.03125  # 0.375 kph
  FWD_BLACKLISTED_ADDRS = {2: [0x50, 0x2a4]}

  MAX_RATE_UP = 2
  MAX_RATE_DOWN = 3
  MAX_TORQUE_LOOKUP = [0], [270]

  MAX_RT_DELTA = 112

  DRIVER_TORQUE_ALLOWANCE = 250
  DRIVER_TORQUE_FACTOR = 2

  # Safety around steering req bit
  MIN_VALID_STEERING_FRAMES = 89
  MAX_INVALID_STEERING_FRAMES = 2

  PT_BUS = 0
  SCC_BUS = 2
  STEER_BUS = 0
  STEER_MSG = ""
  GAS_MSG = ("", "")
  BUTTONS_TX_BUS = 1

  def _torque_driver_msg(self, torque):
    values = {"MDPS_StrTqSnsrVal": torque}
    return self.packer.make_can_msg_safety("MDPS", self.PT_BUS, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"StrTqReqVal": torque, "ActToiSta": steer_req}
    return self.packer.make_can_msg_safety(self.STEER_MSG, self.STEER_BUS, values)

  def _speed_msg(self, speed):
    values = {f"WHL_Spd{pos}Val": speed * 3.6 for pos in ["FL", "FR", "RL", "RR"]}
    return self.packer.make_can_msg_safety("WHEEL_SPEEDS", self.PT_BUS, values)

  def _user_brake_msg(self, brake):
    values = {"DriverBraking": brake}
    return self.packer.make_can_msg_safety("TCS", self.PT_BUS, values)

  def _user_gas_msg(self, gas):
    values = {self.GAS_MSG[1]: gas}
    return self.packer.make_can_msg_safety(self.GAS_MSG[0], self.PT_BUS, values)

  def _pcm_status_msg(self, enable):
    values = {"ACCMode": 1 if enable else 0}
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.SCC_BUS, values)

  def _button_msg(self, buttons, main_button=0, bus=None):
    if bus is None:
      bus = self.PT_BUS
    values = {
      "CRUISE_BUTTONS": buttons,
      "ADAPTIVE_CRUISE_MAIN_BTN": main_button,
    }
    return self.packer.make_can_msg_safety("CRUISE_BUTTONS", bus, values)

  def _acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.SCC_BUS, values)

  def _lkas_button_msg(self, enabled):
    values = {"LDA_BTN": enabled}
    return self.packer.make_can_msg_safety("CRUISE_BUTTONS", self.PT_BUS, values)

  def _main_cruise_button_msg(self, enabled):
    return self._button_msg(0, enabled)


class TestHyundaiCanfdTorqueSteering(TestHyundaiCanfdBase, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ == "TestHyundaiCanfdTorqueSteering":
      cls.packer = None
      raise unittest.SkipTest


class TestHyundaiCanfdAngleSteering(TestHyundaiCanfdBase, common.AngleSteeringSafetyTest):

  SAFETY_PARAM = HyundaiSafetyFlags.CANFD_ANGLE_STEERING

  PLATFORMS = {str(platform): platform for platform in CAR if
               platform.config.flags & HyundaiFlags.CANFD_ANGLE_STEERING and not CarInterface.get_non_essential_params(str(platform)).dashcamOnly}

  # Angle control limits
  BASELINE_PANDA_ANGLE_LIMITS: AngleSteeringLimitsVM = AngleSteeringLimitsVM(
    360,  # degrees (safe upper bound for command, allowing some margin)
    MAX_ANGLE_RATE=5  # comfort rate limit for angle commands, in degrees per frame.
  )

  STEER_ANGLE_MAX = 360  # deg
  DEG_TO_CAN = 10
  ANGLE_SAFETY_THRESHOLD_PCT = -2.0  # Fail if difference is less than -2%

  # Hyundai uses get_max_angle_delta and get_max_angle for real lateral accel and jerk limits
  # TODO: integrate this into AngleSteeringSafetyTest
  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None

  # Real time limits
  LATERAL_FREQUENCY = 100  # Hz

  cnt_angle_cmd = 0

  def get_angle_limits(self, car_name):
    limits = CarControllerParams(CarInterface.get_non_essential_params(car_name))
    limits.ANGLE_LIMITS = self.BASELINE_PANDA_ANGLE_LIMITS
    return limits

  def get_baseline_limits(self):
    return self.get_angle_limits(ANGLE_SAFETY_BASELINE_MODEL)

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True, gain: float = 0.0):
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    values = {"ADAS_StrAnglReqVal": angle, "LKAS_ANGLE_ACTIVE": 2 if enabled else 1,
              "ADAS_ACIAnglTqRedcGainVal": gain}
    return self.packer.make_can_msg_safety(self.STEER_MSG, self.STEER_BUS, values)

  def _angle_meas_msg(self, angle: float):
    values = {"MDPS_EstStrAnglVal": angle}
    return self.packer.make_can_msg_safety("MDPS", self.PT_BUS, values)

  def _get_steer_cmd_angle_max(self, speed):
    baseline_vm = self.get_vm(ANGLE_SAFETY_BASELINE_MODEL)
    return get_max_angle_vm(max(speed, 1), baseline_vm, self.get_baseline_limits())

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ == "TestHyundaiCanfdAngleSteering":
      cls.packer = None
      raise unittest.SkipTest

  def get_vm(self, car_name):
    return VehicleModel(CarInterface.get_non_essential_params(car_name))

  def test_angle_cmd_when_enabled(self):
    # We properly test lateral acceleration and jerk below
    pass

  def test_lateral_accel_limit(self):
    car_name = ANGLE_SAFETY_BASELINE_MODEL
    for speed in np.linspace(0, 40, 100):
      speed = round_speed(away_round(speed / 0.03125 * 3.6) * 0.03125 / 3.6)
      speed = max(speed, 1)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed

        # at limit (safety tolerance adds 1)
        angl = get_max_angle_vm(speed, self.get_vm(car_name), self.get_angle_limits(car_name))
        max_angle = round_angle(get_max_angle_vm(speed, self.get_vm(car_name), self.get_angle_limits(car_name)), 1) * sign
        max_angle = np.clip(max_angle, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self.safety.set_desired_angle_last(round(max_angle * self.DEG_TO_CAN))

        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle, True)), f"{angl} -- {max_angle}")

        # above limit (offset 6 to reliably exceed C float tolerance)
        max_angle_raw = round_angle(get_max_angle_vm(speed, self.get_vm(car_name), self.get_angle_limits(car_name)), 6) * sign
        max_angle = np.clip(max_angle_raw, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self._tx(self._angle_cmd_msg(max_angle, True))

        # at low speeds max angle is above 360, so adding 1 has no effect
        should_tx = abs(max_angle_raw) >= self.STEER_ANGLE_MAX
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(max_angle, True)), f"should_tx: {should_tx}, max_angle: {max_angle}, speed: {speed}")

  @parameterized("car_name", sorted(PLATFORMS))
  def test_lateral_jerk_limit(self, car_name):
    # car_name = ANGLE_SAFETY_BASELINE_MODEL
    angle_model_id = ANGLE_STEERING_MODEL_BY_CAR.get(car_name, 0)
    self.safety.set_current_safety_param_sp(encode_angle_model_id(angle_model_id))
    self._reset_safety_hooks()

    for speed in np.linspace(0, 40, 100):
      speed = round_speed(away_round(speed / 0.03125 * 3.6) * 0.03125 / 3.6)
      speed = max(speed, 1)
      for sign in (-1, 1):  # (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed
        self._tx(self._angle_cmd_msg(0, True))

        # Stay within limits
        # Up
        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.get_vm(car_name), self.get_angle_limits(car_name))) * sign
        sent = self._tx(self._angle_cmd_msg(max_angle_delta, True))
        self.assertTrue(sent)

        # Don't change
        self.safety.set_desired_angle_last(round(max_angle_delta * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        # Inject too high rates
        # Up
        # TODO-SP: Why do I need to set a can_offset so high to pass the tests and why tesla only does +1? and why does it seem to differ based on the baseline?
        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.get_vm(car_name), self.get_angle_limits(car_name)), 6) * sign
        self.assertFalse(self._tx(self._angle_cmd_msg(max_angle_delta, True)), vars(self.get_angle_limits(car_name)))

        # Don't change
        self.safety.set_desired_angle_last(round(max_angle_delta * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))

        # Down
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))

        # Recover
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

  def test_rt_limits(self):
    # TODO: remove and check all safety modes
    if self.LATERAL_FREQUENCY == -1:
      raise unittest.SkipTest("No real time limits")

    # Angle safety enforces real time limits by checking the message send frequency in a 250ms time window
    self.safety.set_timer(0)
    self.safety.set_controls_allowed(True)
    max_rt_msgs = int(self.LATERAL_FREQUENCY * common.RT_INTERVAL / 1e6 * 1.2 + 1)  # 1.2x buffer

    for i in range(max_rt_msgs * 2):
      should_tx = i <= max_rt_msgs
      self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))

    # One under RT interval should do nothing
    self.safety.set_timer(common.RT_INTERVAL - 1)
    for _ in range(5):
      self.assertFalse(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))

    # Increment timer and send 1 message to reset RT window
    self.safety.set_timer(common.RT_INTERVAL)
    self.assertFalse(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))
    for _ in range(5):
      self.assertTrue(self._tx(self._angle_cmd_msg(0, True, increment_timer=False)))

  def test_torque_reduction_gain(self):
    # Valid gains when enabled
    for gain in [0.0, 0.5, 1.0]:
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._tx(self._angle_cmd_msg(0, True, gain=gain)),
                      f"gain={gain} should be allowed when enabled")

    # Reserved values (raw 251+) must fail even when enabled
    for gain in [1.004, 1.008, 1.02]:
      self.safety.set_controls_allowed(True)
      self.assertFalse(self._tx(self._angle_cmd_msg(0, True, gain=gain)),
                       f"gain={gain} (reserved) should be blocked")

    # Non-zero gain when disabled must fail
    for gain in [0.004, 0.5, 1.0]:
      self.safety.set_controls_allowed(True)
      self.assertFalse(self._tx(self._angle_cmd_msg(0, False, gain=gain)),
                       f"gain={gain} should be blocked when disabled")

    # Zero gain when disabled must pass
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._angle_cmd_msg(0, False, gain=0.0)))

  @parameterized("car_name", sorted(PLATFORMS))
  def test_max_steering_angle_safety(self, car_name):
    """
    Test that ensures the current car's max steering angles are never more than 2%
    lower than the baseline car across all test speeds.
    """
    baseline_car = ANGLE_SAFETY_BASELINE_MODEL
    baseline_vm = self.get_vm(baseline_car)
    current_vm = self.get_vm(car_name)

    for speed in np.linspace(1, 40, 10):
      baseline_max_angle = get_max_angle_vm(speed, baseline_vm, self.get_baseline_limits())
      current_max_angle = get_max_angle_vm(speed, current_vm, self.get_angle_limits(car_name))

      # Skip if both exceed STEER_ANGLE_MAX (only_relevant_angles logic)
      if current_max_angle > self.STEER_ANGLE_MAX and baseline_max_angle > self.STEER_ANGLE_MAX:
        continue

      # Calculate percentage difference
      if baseline_max_angle != 0:
        angle_diff_pct = ((current_max_angle - baseline_max_angle) / baseline_max_angle) * 100
      else:
        angle_diff_pct = 0

      # Assert that difference is not dangerously low
      self.assertTrue(
        angle_diff_pct >= self.ANGLE_SAFETY_THRESHOLD_PCT,
        f"{car_name} max steering angle at {speed:.1f} m/s [{current_max_angle:.2f}°] is {angle_diff_pct:.2f}% " +
        f"lower than baseline {baseline_car} ({current_max_angle:.2f}° vs {baseline_max_angle:.2f}°). " +
        f"Must be >= {self.ANGLE_SAFETY_THRESHOLD_PCT}% to ensure safety." +
        f"Consider updating the baseline model to be {car_name} (which will lower the threshold for ALL models). " +
        f"Slip Factor: {repr(calc_slip_factor(current_vm))}"
      )

  @parameterized("car_name", sorted(PLATFORMS))
  def test_max_steering_angle_delta_safety(self, car_name):
    """
    Test that ensures the current car's max steering angle deltas are never more than 2%
    lower than the baseline car across all test speeds.
    """
    baseline_car = ANGLE_SAFETY_BASELINE_MODEL
    baseline_vm = self.get_vm(baseline_car)
    baseline_limits = CarControllerParams(CarInterface.get_non_essential_params(baseline_car))
    current_vm = self.get_vm(car_name)
    current_limits = CarControllerParams(CarInterface.get_non_essential_params(car_name))

    for speed in np.linspace(1, 40, 10):
      baseline_max_delta = get_max_angle_delta_vm(speed, baseline_vm, baseline_limits)
      current_max_delta = get_max_angle_delta_vm(speed, current_vm, current_limits)

      # Calculate percentage difference
      if baseline_max_delta != 0:
        delta_diff_pct = ((current_max_delta - baseline_max_delta) / baseline_max_delta) * 100
      else:
        delta_diff_pct = 0

      # Assert that difference is not dangerously low
      self.assertTrue(
        delta_diff_pct >= self.ANGLE_SAFETY_THRESHOLD_PCT,
        f"{car_name} max steering angle delta at {speed:.1f} m/s is {delta_diff_pct:.2f}% " +
        f"lower than {baseline_car} ({current_max_delta:.4f} vs {baseline_max_delta:.4f} deg/frame). " +
        f"Must be >= {self.ANGLE_SAFETY_THRESHOLD_PCT}% to ensure safety." +
        f"Consider updating the baseline model to be {car_name} (which will lower the threshold for ALL models)." +
        f"Slip Factor: {repr(calc_slip_factor(current_vm))}"
      )

  @parameterized("car_name", sorted(PLATFORMS))
  def test_valid_angle_model_extraction(self, car_name):
    """
    Ensures that the safety layer correctly extracts and assigns the expected
    angle steering model ID from the safety_param_sp bitmask for every supported platform.
    """
    angle_model_id = ANGLE_STEERING_MODEL_BY_CAR.get(car_name, 0)
    self.safety.set_current_safety_param_sp(encode_angle_model_id(angle_model_id))
    self._reset_safety_hooks()
    self.assertEqual(self.safety.get_hyundai_angle_model_id(), angle_model_id)

  def test_out_of_bounds_angle_model_fallback(self):
    """
    Verifies that passing an unknown or out of bounds angle model ID
    via safety_param_sp safely defaults to the baseline fallback model (ID 0).
    """
    model_count = len(ANGLE_STEERING_MODEL_BY_CAR)
    self.safety.set_current_safety_param_sp(encode_angle_model_id(model_count + 1))
    self._reset_safety_hooks()
    self.assertEqual(self.safety.get_hyundai_angle_model_id(), 0)

  def test_angle_model_bitmask_isolation(self):
    """
    Tests that bits [0:3] and [8:15] in safety_param_sp are properly ignored.
    Verifies the shift and mask logic works regardless of other flags being set.
    """
    test_id = 2  # HYUNDAI_IONIQ_5_PE

    # 0xFF0F sets every bit to 1 EXCEPT bits 4-7.
    noise_mask = 0xFF0F
    param_with_noise = encode_angle_model_id(test_id) | noise_mask

    self.safety.set_current_safety_param_sp(param_with_noise)
    self._reset_safety_hooks()

    self.assertEqual(self.safety.get_hyundai_angle_model_id(), test_id)


class TestHyundaiCanfdLFASteeringBase(TestHyundaiCanfdTorqueSteering):

  TX_MSGS = [[0x12A, 0], [0x1A0, 1], [0x1CF, 0], [0x1E0, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x12A, 0x1E0)}  # LFA, LFAHDA_CLUSTER
  FWD_BLACKLISTED_ADDRS = {2: [0x12A, 0x1E0]}

  STEER_MSG = "LFA"
  BUTTONS_TX_BUS = 2
  EXTRA_SAFETY_PARAM: int

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ in ("TestHyundaiCanfdLFASteering", "TestHyundaiCanfdLFASteeringAltButtons"):
      cls.packer = None
      raise unittest.SkipTest

  @property
  def SAFETY_PARAM(self):
    return self.EXTRA_SAFETY_PARAM
@parameterized_class(ALL_GAS_EV_HYBRID_COMBOS)
class TestHyundaiCanfdLFASteering(TestHyundaiCanfdLFASteeringBase):
  pass
class TestHyundaiCanfdLFASteeringAltButtonsBase(TestHyundaiCanfdLFASteeringBase):

  EXTRA_SAFETY_PARAM: int

  @property
  def SAFETY_PARAM(self):
    return HyundaiSafetyFlags.CANFD_ALT_BUTTONS | self.EXTRA_SAFETY_PARAM

  def _button_msg(self, buttons, main_button=0, bus=1):
    values = {
      "CRUISE_BUTTONS": buttons,
      "ADAPTIVE_CRUISE_MAIN_BTN": main_button,
    }
    return self.packer.make_can_msg_safety("CRUISE_BUTTONS_ALT", self.PT_BUS, values)

  def _lkas_button_msg(self, enabled):
    values = {"LDA_BTN": enabled}
    return self.packer.make_can_msg_safety("CRUISE_BUTTONS_ALT", self.PT_BUS, values)

  def _acc_cancel_msg(self, cancel, accel=0):
    values = {"ACCMode": 4 if cancel else 0, "aReqRaw": accel, "aReqValue": accel}
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.PT_BUS, values)

  def test_button_sends(self):
    """
      No button send allowed with alt buttons.
    """
    for enabled in (True, False):
      for btn in range(8):
        self.safety.set_controls_allowed(enabled)
        self.assertFalse(self._tx(self._button_msg(btn)))

  def test_acc_cancel(self):
    # FIXME: the CANFD_ALT_BUTTONS cars are the only ones that use SCC_CONTROL to cancel, why can't we use buttons?
    for enabled in (True, False):
      self.safety.set_controls_allowed(enabled)
      self.assertTrue(self._tx(self._acc_cancel_msg(True)))
      self.assertFalse(self._tx(self._acc_cancel_msg(True, accel=1)))
      self.assertFalse(self._tx(self._acc_cancel_msg(False)))
@parameterized_class(ALL_GAS_EV_HYBRID_COMBOS)
class TestHyundaiCanfdLFASteeringAltButtons(TestHyundaiCanfdLFASteeringAltButtonsBase):
  pass


class TestHyundaiCanfdLKASteeringEV(TestHyundaiCanfdTorqueSteering):

  SAFETY_PARAM = HyundaiSafetyFlags.CANFD_LKA_STEER_MSG | HyundaiSafetyFlags.EV_GAS

  TX_MSGS = [[0x50, 0], [0x1CF, 1], [0x2A4, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x50, 0x2a4)}  # LKAS, CAM_0x2A4
  FWD_BLACKLISTED_ADDRS = {2: [0x50, 0x2a4]}

  PT_BUS = 1
  SCC_BUS = 1
  STEER_MSG = "LKAS"
  GAS_MSG = ("ACCELERATOR", "ACCELERATOR_PEDAL")
# TODO: Handle ICE and HEV configurations once we see cars that use the new messages
class TestHyundaiCanfdLKASteeringAltEVBase(TestHyundaiCanfdBase):

  SAFETY_PARAM = HyundaiSafetyFlags.CANFD_LKA_STEER_MSG | HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CANFD_LKA_STEER_MSG_ALT

  TX_MSGS = [[0x110, 0], [0x1CF, 1], [0x362, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x110, 0x362)}  # LKAS_ALT, CAM_0x362
  FWD_BLACKLISTED_ADDRS = {2: [0x110, 0x362]}

  PT_BUS = 1
  SCC_BUS = 1
  STEER_MSG = "LKAS_ALT"
  GAS_MSG = ("ACCELERATOR", "ACCELERATOR_PEDAL")


class TestHyundaiCanfdLKASteeringAltEVTorque(TestHyundaiCanfdLKASteeringAltEVBase, TestHyundaiCanfdTorqueSteering):
  pass


class TestHyundaiCanfdLKASteeringAltAngle(TestHyundaiCanfdAngleSteering):

  SAFETY_PARAM = HyundaiSafetyFlags.CANFD_LKA_STEER_MSG | HyundaiSafetyFlags.CANFD_LKA_STEER_MSG_ALT | HyundaiSafetyFlags.CANFD_ANGLE_STEERING

  TX_MSGS = [[0x110, 0], [0x1CF, 1], [0x362, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x110, 0x362)}
  FWD_BLACKLISTED_ADDRS = {2: [0x110, 0x362]}

  PT_BUS = 1
  SCC_BUS = 1
  STEER_MSG = "LKAS_ALT"
  GAS_MSG = ("ACCELERATOR_BRAKE_ALT", "ACCELERATOR_PEDAL_PRESSED")

  # Angle steering does not use torque — override inherited torque tests
  def test_steer_safety_check(self):
    pass

  def test_non_realtime_limit_up(self):
    pass

  def test_steer_req_bit(self):
    pass

  def test_steer_req_bit_frames(self):
    pass

  def test_steer_req_bit_multi_invalid(self):
    pass

  def test_steer_req_bit_realtime(self):
    pass

  def test_against_torque_driver(self):
    pass

  def test_realtime_limits(self):
    pass


class TestHyundaiCanfdLFASteeringAngleLFAAlt(TestHyundaiCanfdAngleSteering):
  # HDA1 + LFA2 cars that steer by angle through LFA_ALT (0xCB) instead of LFA (0x12A)

  TX_MSGS = [[0x12A, 0], [0x1A0, 1], [0x1CF, 0], [0x1E0, 0], [0xCB, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x12A, 0x1E0)}
  FWD_BLACKLISTED_ADDRS = {2: [0x12A, 0x1E0, 0xCB]}

  STEER_MSG = "LFA_ALT"
  BUTTONS_TX_BUS = 2
  SCC_BUS = 0
  GAS_MSG = ("ACCELERATOR", "ACCELERATOR_PEDAL")

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_canfd_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, HyundaiSafetyFlags.EV_GAS |
                                 HyundaiSafetyFlags.CANFD_ANGLE_STEERING | HyundaiSafetyFlags.CANFD_LFA_ALT_STEERING)
    self.safety.init_tests()

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True, gain: float = 0.0):
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    values = {"ADAS_StrAnglReqVal": angle, "ADAS_ActvACILvl2Sta": 2 if enabled else 1,
              "ADAS_ACIAnglTqRedcGainVal": gain}
    return self.packer.make_can_msg_safety("LFA_ALT", 0, values)

  def test_lfa_alt_requires_flag(self):
    # without HYUNDAI_PARAM_CANFD_LFA_ALT_STEERING, sending 0xCB must be rejected
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd,
                                 HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CANFD_ANGLE_STEERING)
    self.safety.init_tests()
    self.assertFalse(self._tx(self._angle_cmd_msg(0, False)))

  # Angle steering does not use torque — override inherited torque tests
  def test_steer_safety_check(self):
    pass

  def test_non_realtime_limit_up(self):
    pass

  def test_steer_req_bit(self):
    pass

  def test_steer_req_bit_frames(self):
    pass

  def test_steer_req_bit_multi_invalid(self):
    pass

  def test_steer_req_bit_realtime(self):
    pass

  def test_against_torque_driver(self):
    pass

  def test_realtime_limits(self):
    pass


class TestHyundaiCanfdLKASteeringLongEV(HyundaiLongitudinalBase, TestHyundaiCanfdLKASteeringEV):

  SAFETY_PARAM = HyundaiSafetyFlags.CANFD_LKA_STEER_MSG | HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.EV_GAS

  TX_MSGS = [[0x50, 0], [0x1CF, 1], [0x2A4, 0], [0x51, 0], [0x730, 1], [0x12a, 1], [0x160, 1],
             [0x1e0, 1], [0x1a0, 1], [0x1ea, 1], [0x200, 1], [0x345, 1], [0x1da, 1]]

  RELAY_MALFUNCTION_ADDRS = {0: (0x50, 0x2a4), 1: (0x1a0,)}  # LKAS, CAM_0x2A4, SCC_CONTROL

  DISABLED_ECU_UDS_MSG = (0x730, 1)
  DISABLED_ECU_ACTUATION_MSG = (0x1a0, 1)

  STEER_MSG = "LFA"
  GAS_MSG = ("ACCELERATOR", "ACCELERATOR_PEDAL")
  STEER_BUS = 1

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel,
    }
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.PT_BUS, values)

  def _tx_acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.PT_BUS, values)
# Tests longitudinal for ICE, hybrid, EV cars with LFA steering
class TestHyundaiCanfdLFASteeringLongBase(HyundaiLongitudinalBase, TestHyundaiCanfdLFASteeringBase):

  FWD_BLACKLISTED_ADDRS = {2: [0x12a, 0x1e0, 0x1a0, 0x160]}

  RELAY_MALFUNCTION_ADDRS = {0: (0x12A, 0x1E0, 0x1a0, 0x160)}  # LFA, LFAHDA_CLUSTER, SCC_CONTROL, ADRV_0x160

  DISABLED_ECU_UDS_MSG = (0x7D0, 0)
  DISABLED_ECU_ACTUATION_MSG = (0x1a0, 0)

  @property
  def SAFETY_PARAM(self):
    return HyundaiSafetyFlags.LONG | self.EXTRA_SAFETY_PARAM

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel,
    }
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.PT_BUS, values)

  def _tx_acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC_CONTROL", self.PT_BUS, values)

  def test_tester_present_allowed(self, ecu_disable: bool = True):
    super().test_tester_present_allowed(ecu_disable=not self.EXTRA_SAFETY_PARAM & HyundaiSafetyFlags.CAMERA_SCC)
@parameterized_class(ALL_GAS_EV_HYBRID_COMBOS)
class TestHyundaiCanfdLFASteeringLong(TestHyundaiCanfdLFASteeringLongBase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ == "TestHyundaiCanfdLFASteeringLong":
      raise unittest.SkipTest
@parameterized_class(ALL_GAS_EV_HYBRID_COMBOS)
class TestHyundaiCanfdLFASteeringLongAltButtons(TestHyundaiCanfdLFASteeringLongBase, TestHyundaiCanfdLFASteeringAltButtonsBase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ == "TestHyundaiCanfdLFASteeringLongAltButtons":
      raise unittest.SkipTest

  @property
  def SAFETY_PARAM(self):
    return HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CANFD_ALT_BUTTONS | self.EXTRA_SAFETY_PARAM

  def test_acc_cancel(self):
    # Alt buttons does not use SCC_CONTROL to cancel if longitudinal
    pass


class HyundaiCanfdCCNCTest:

  CCNC_SAFETY_PARAM = HyundaiSafetyFlags.CCNC

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.__name__ in (
      "TestHyundaiCanfdLFASteeringCCNC",
      "TestHyundaiCanfdLFASteeringLongCCNC",
    ):
      cls.packer = None
      raise unittest.SkipTest

  @property
  def SAFETY_PARAM(self):
    return self.CCNC_SAFETY_PARAM | super().SAFETY_PARAM

  def test_ccnc_tx_msgs(self):
    for msg, bus in (("CCNC_0x161", 0), ("CCNC_0x162", 0), ("MDPS", 2)):
      self.assertTrue(self._tx(self.packer.make_can_msg_safety(msg, bus, {})))
    self.assertTrue(self._tx(common.make_msg(2, 0x7C4)))


@parameterized_class(CAMERA_SCC_COMBOS)
class TestHyundaiCanfdLFASteeringCCNC(HyundaiCanfdCCNCTest, TestHyundaiCanfdLFASteeringBase):

  TX_MSGS = [[0x12A, 0], [0x1E0, 0], [0x1CF, 2], [0x7C4, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x12A, 0x1E0, 0x161, 0x162), 2: (0x7C4, 0xEA)}
  FWD_BLACKLISTED_ADDRS = {2: [0x12A, 0x1E0, 0x161, 0x162], 0: [0x7C4, 0xEA]}


@parameterized_class(CAMERA_SCC_COMBOS)
class TestHyundaiCanfdLFASteeringLongCCNC(HyundaiCanfdCCNCTest, TestHyundaiCanfdLFASteeringLongBase):

  CCNC_SAFETY_PARAM = HyundaiSafetyFlags.CCNC | HyundaiSafetyFlags.LONG

  TX_MSGS = [[0x12A, 0], [0x1E0, 0], [0x1CF, 2], [0x7C4, 2], [0x1A0, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x12A, 0x1E0, 0x161, 0x162, 0x1A0), 2: (0x7C4, 0xEA)}
  FWD_BLACKLISTED_ADDRS = {2: [0x12A, 0x1E0, 0x161, 0x162, 0x1A0], 0: [0x7C4, 0xEA]}


class TestHyundaiCanfdEV3Gen(unittest.TestCase):
  """KIA_EV3-generation safety branches: CCNC + angle steering + alt buttons + camera SCC + EV gas.

  The parameterized CCNC classes are skipped, so the EV3-specific branches (main-cruise toggle
  ownership, the longitudinal RX check variant, SCC_CONTROL semantics per mode) are covered here
  directly against libsafety.
  """

  EV3_PARAM = int(HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CAMERA_SCC | HyundaiSafetyFlags.CANFD_ALT_BUTTONS |
                  HyundaiSafetyFlags.CANFD_ANGLE_STEERING | HyundaiSafetyFlags.CCNC | HyundaiSafetyFlags.CANFD_LFA_ALT_STEERING)

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_canfd_generated")
    self.safety = libsafety_py.libsafety

  def _init(self, long, main_cruise_toggleable=True):
    self.safety.set_current_safety_param_sp(int(HyundaiSafetyFlagsSP.LONG_MAIN_CRUISE_TOGGLEABLE) if main_cruise_toggleable else 0)
    param = self.EV3_PARAM | int(HyundaiSafetyFlags.LONG) if long else self.EV3_PARAM
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param)
    self.safety.init_tests()

  def _rx(self, name, values, bus=0):
    return self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, bus, values))

  def _tx(self, name, values, bus=0):
    return self.safety.safety_tx_hook(self.packer.make_can_msg_safety(name, bus, values))

  def _rx_wheel_buttons(self, code, lfa=0):
    # EV3 steering wheel buttons: STEERING_BUTTONS_ALT (0x10B), byte 10 low nibble
    self.assertTrue(self._rx("STEERING_BUTTONS_ALT", {"CRUISE_BUTTONS": code, "LFA_BTN": lfa}))

  def _press_main(self):
    self._rx_wheel_buttons(8)
    self._rx_wheel_buttons(0)

  def test_main_toggle_single_edge(self):
    # exactly one toggle per rising edge, in every mode/param combination
    for long in (False, True):
      for toggleable in (False, True):
        with self.subTest(long=long, main_cruise_toggleable=toggleable):
          self._init(long, toggleable)
          self.assertFalse(self.safety.get_acc_main_on())
          self._rx_wheel_buttons(8)
          self.assertTrue(self.safety.get_acc_main_on())
          # held button must not re-toggle
          self._rx_wheel_buttons(8)
          self.assertTrue(self.safety.get_acc_main_on())
          self._rx_wheel_buttons(0)
          self._press_main()
          self.assertFalse(self.safety.get_acc_main_on())

  def test_long_button_engage(self):
    self._init(long=True)
    self.assertFalse(self.safety.get_controls_allowed())
    self._rx_wheel_buttons(2)  # SET
    self._rx_wheel_buttons(0)  # falling edge
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx_wheel_buttons(4)  # CANCEL
    self.assertFalse(self.safety.get_controls_allowed())

  def test_long_zeroed_alt_buttons_ignored(self):
    # EV3 broadcasts CRUISE_BUTTONS_ALT with the button fields zeroed; even if it carried button
    # data it must not feed engage logic or interleave zeros into the 0x10B edge detection
    self._init(long=True)
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": 2, "ADAPTIVE_CRUISE_MAIN_BTN": 0}))
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": 0, "ADAPTIVE_CRUISE_MAIN_BTN": 0}))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": 0, "ADAPTIVE_CRUISE_MAIN_BTN": 1}))
    self.assertFalse(self.safety.get_acc_main_on())
    # held SET on 0x10B with zeroed 0x1AA frames in between must not fake falling edges
    self._rx_wheel_buttons(2)
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": 0, "ADAPTIVE_CRUISE_MAIN_BTN": 0}))
    self._rx_wheel_buttons(2)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_long_accel_limits(self):
    self._init(long=True)
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 2.0, "aReqValue": 2.0}))
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": -3.5, "aReqValue": -3.5}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 2.1, "aReqValue": 2.1}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": -3.6, "aReqValue": -3.6}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 0.0, "aReqValue": 2.5}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 2.5, "aReqValue": 0.0}))

  def test_long_disengaged_accel_gating(self):
    self._init(long=True)
    self.safety.set_controls_allowed(False)
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 0.01, "aReqValue": 0.01}))
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 0, "aReqRaw": 0, "aReqValue": 0}))
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 4, "aReqRaw": 0, "aReqValue": 0}))

  def test_long_gas_override_blocks_longitudinal(self):
    self._init(long=True)
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._rx("ACCELERATOR", {"ACCELERATOR_PEDAL": 100}))
    self.assertTrue(self.safety.get_controls_allowed())  # lateral stays engaged
    self.assertFalse(self.safety.get_longitudinal_allowed())
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 2, "aReqRaw": 1.0, "aReqValue": 1.0}))
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 2, "aReqRaw": 0, "aReqValue": 0}))
    self.assertTrue(self._rx("ACCELERATOR", {"ACCELERATOR_PEDAL": 0}))
    self.assertTrue(self.safety.get_longitudinal_allowed())

  def test_long_rx_whitelist(self):
    self._init(long=True)
    # EV3 quirk messages must be accepted by the longitudinal RX checks
    self.assertTrue(self._rx("ACCELERATOR", {"ACCELERATOR_PEDAL": 0}))
    self.assertTrue(self._rx("STEERING_BUTTONS_ALT", {"LFA_BTN": 1}))
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": 0, "ADAPTIVE_CRUISE_MAIN_BTN": 0}))

  def test_long_forwarding(self):
    self._init(long=True)
    for addr in (0x1A0, 0x161, 0x162, 0xCB):
      self.assertEqual(self.safety.safety_fwd_hook(2, addr), -1, f"stock 0x{addr:X} not blocked")
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1B5), 0)
    self.assertEqual(self.safety.safety_fwd_hook(0, 0x1AA), 2)

  def test_stock_scc_semantics(self):
    self._init(long=False)
    # stock mode: SCC_CONTROL TX only as zero-accel cancel, camera SCC forwarded
    self.assertTrue(self._tx("SCC_CONTROL", {"ACCMode": 4, "aReqRaw": 0, "aReqValue": 0}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 1, "aReqRaw": 0, "aReqValue": 0}))
    self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 4, "aReqRaw": 1.0, "aReqValue": 1.0}))
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1A0), 0)

  def test_reinit_clears_state(self):
    self._init(long=True)
    self._rx_wheel_buttons(8)
    self.assertTrue(self.safety.get_acc_main_on())
    self._init(long=False)
    self.assertFalse(self.safety.get_acc_main_on())
    self.assertFalse(self.safety.get_controls_allowed())


class TestHyundaiCanfdEV3GenLKASteering(unittest.TestCase):
  """KIA_EV3-generation HDA2 trim (KIA_EV4): LKA steering through LKAS_ALT on the relayed A-CAN while the
  E-CAN quirks (counter-less ACCELERATOR, buttons in STEERING_BUTTONS_ALT, zeroed CRUISE_BUTTONS_ALT, no
  CRUISE_BUTTONS) sit on the bus 1 tap. Lateral only: LKA steering angle cars get no openpilot longitudinal,
  stock SCC comes from the ADAS ECU on bus 1.
  """

  PARAM = int(HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CANFD_LKA_STEER_MSG | HyundaiSafetyFlags.CANFD_LKA_STEER_MSG_ALT |
              HyundaiSafetyFlags.CANFD_ALT_BUTTONS | HyundaiSafetyFlags.CANFD_ANGLE_STEERING | HyundaiSafetyFlags.CCNC |
              HyundaiSafetyFlags.CANFD_LFA_ALT_STEERING)
  PT_BUS = 1  # E-CAN tap
  A_BUS = 0   # relayed A-CAN, car side

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_canfd_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(encode_angle_model_id(ANGLE_STEERING_MODEL_BY_CAR[str(CAR.KIA_EV4)]))
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, self.PARAM)
    self.safety.init_tests()
    self.cnt_angle_cmd = 0

  def _rx(self, name, values, bus=PT_BUS):
    return self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, bus, values))

  def _tx(self, name, values, bus):
    return self.safety.safety_tx_hook(self.packer.make_can_msg_safety(name, bus, values))

  def _rx_wheel_buttons(self, code, lfa=0):
    self.assertTrue(self._rx("STEERING_BUTTONS_ALT", {"CRUISE_BUTTONS": code, "LFA_BTN": lfa}))

  def _angle_cmd(self, angle, active, bus=A_BUS):
    self.safety.set_timer(self.cnt_angle_cmd * 10000)
    self.cnt_angle_cmd += 1
    values = {"ADAS_StrAnglReqVal": angle, "LKAS_ANGLE_ACTIVE": 2 if active else 1, "ADAS_ACIAnglTqRedcGainVal": 0}
    return self._tx("LKAS_ALT", values, bus)

  def test_rx_checks_on_ecan_tap(self):
    # the EV3-generation RX check set on bus 1 is complete without CRUISE_BUTTONS (0x1cf)
    for name in ("ACCELERATOR", "TCS", "WHEEL_SPEEDS", "MDPS", "CRUISE_BUTTONS_ALT", "STEERING_BUTTONS_ALT", "SCC_CONTROL"):
      self.assertTrue(self._rx(name, {}), name)
    self.assertTrue(self.safety.safety_config_valid())

  def test_accelerator_counter_ignored(self):
    # EV3-generation ACCELERATOR (0x35) does not count +1 per frame
    for _ in range(common.MAX_WRONG_COUNTERS + 2):
      self.assertTrue(self._rx("ACCELERATOR", {"COUNTER": 3}))

  def test_main_toggle_from_wheel_buttons(self):
    self.assertFalse(self.safety.get_acc_main_on())
    self._rx_wheel_buttons(8)
    self.assertTrue(self.safety.get_acc_main_on())
    self._rx_wheel_buttons(8)  # held button must not re-toggle
    self.assertTrue(self.safety.get_acc_main_on())
    self._rx_wheel_buttons(0)
    # the zeroed CRUISE_BUTTONS_ALT stream carries no main state here
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"ADAPTIVE_CRUISE_MAIN_BTN": 1}))
    self.assertTrue(self.safety.get_acc_main_on())
    self._rx_wheel_buttons(8)
    self._rx_wheel_buttons(0)
    self.assertFalse(self.safety.get_acc_main_on())

  def test_engage_from_wheel_buttons_and_stock_scc(self):
    # stock cruise engaging on the bus 1 tap right after SET on STEERING_BUTTONS_ALT allows controls
    self.assertTrue(self._rx("SCC_CONTROL", {"ACCMode": 0}))
    self.assertFalse(self.safety.get_controls_allowed())
    self._rx_wheel_buttons(Buttons.SET)
    self._rx_wheel_buttons(Buttons.NONE)
    self.assertTrue(self._rx("SCC_CONTROL", {"ACCMode": 1}))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._rx("SCC_CONTROL", {"ACCMode": 0}))
    self.assertFalse(self.safety.get_controls_allowed())
    # a SET in the zeroed CRUISE_BUTTONS_ALT stream must not count as a button press (nor age the window)
    for _ in range(PREV_BUTTON_SAMPLES):
      self._rx_wheel_buttons(Buttons.NONE)
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": Buttons.SET}))
    self.assertTrue(self._rx("CRUISE_BUTTONS_ALT", {"CRUISE_BUTTONS": Buttons.NONE}))
    self.assertTrue(self._rx("SCC_CONTROL", {"ACCMode": 1}))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_cruise_buttons_tx_on_ecan_tap(self):
    # stock cruise cancel/resume goes out as CRUISE_BUTTONS (0x1cf) on the bus 1 tap like the other HDA2 cars
    def btn(b, bus=self.PT_BUS):
      return self._tx("CRUISE_BUTTONS", {"CRUISE_BUTTONS": b}, bus)

    self.safety.set_controls_allowed(False)
    self.assertFalse(btn(Buttons.RESUME))
    self.assertFalse(btn(Buttons.SET))
    self.safety.set_controls_allowed(True)
    self.assertTrue(btn(Buttons.RESUME))
    self.assertTrue(btn(Buttons.SET))
    self.assertFalse(btn(Buttons.RESUME, bus=self.A_BUS))
    for engaged in (True, False):
      self.assertTrue(self._rx("SCC_CONTROL", {"ACCMode": 1 if engaged else 0}))
      self.assertEqual(engaged, btn(Buttons.CANCEL))
    # the ADAS ECU owns SCC_CONTROL on the tap: no forged cancel frame on either bus
    for bus in (self.A_BUS, self.PT_BUS):
      self.assertFalse(self._tx("SCC_CONTROL", {"ACCMode": 4, "aReqRaw": 0, "aReqValue": 0}, bus))

  def test_angle_steering_on_acan(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_desired_angle_last(0)
    self.assertTrue(self._angle_cmd(0, True))
    self.assertFalse(self._angle_cmd(0, True, bus=self.PT_BUS))
    self.safety.set_controls_allowed(False)
    self.assertFalse(self._angle_cmd(0, True))
    self.assertTrue(self._angle_cmd(0, False))
    # LFA_ALT stays the ADAS ECU's job on this trim
    for bus in (self.A_BUS, self.PT_BUS):
      self.assertFalse(self._tx("LFA_ALT", {"ADAS_ActvACILvl2Sta": 0, "ADAS_StrAnglReqVal": 0}, bus))

  def test_forwarding(self):
    # the camera's LKAS_ALT / CAM_0x362 are replaced by ours, the rest of A-CAN passes through; the tap is never forwarded
    for addr in (0x110, 0x362):
      self.assertEqual(self.safety.safety_fwd_hook(2, addr), -1, f"camera 0x{addr:X} not blocked")
    for addr in (0x100, 0x165, 0x2F0):
      self.assertEqual(self.safety.safety_fwd_hook(2, addr), 0)
    self.assertEqual(self.safety.safety_fwd_hook(0, 0x3A0), 2)
    self.assertEqual(self.safety.safety_fwd_hook(1, 0x1A0), -1)

  def test_gas_on_ecan_tap(self):
    # gas is read from the EV ACCELERATOR message on the tap; lateral stays engaged, longitudinal is blocked
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._rx("ACCELERATOR", {"ACCELERATOR_PEDAL": 100}))
    self.assertTrue(self.safety.get_gas_pressed_prev())
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    self.assertTrue(self._rx("ACCELERATOR", {"ACCELERATOR_PEDAL": 0}))
    self.assertFalse(self.safety.get_gas_pressed_prev())


if __name__ == "__main__":
  unittest.main()
