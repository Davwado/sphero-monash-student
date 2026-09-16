"""Standalone, low-latency Sphero link - bypasses sphero_unsw's per-command
ack-wait and fixed inter-command sleep for drive commands.

This does not touch any existing lab code, robot.py, or SpheroEduAPI - it's
an alternative you opt into. Two ways to use it:

1. Directly: `fast_drive()` / `get_location()` etc, for a fully custom loop
   (see benchmark.py, benchmark_lab1_move.py).
2. As a drop-in for `Robot`'s `api` argument (see "SpheroEduAPI-compatible
   surface" below and `fast_managed_api()`) - this lets labs 1/2/3 use the
   existing Robot/controller.py/EKF.py/Planner.py completely unmodified,
   just running through the fast comms path instead of SpheroEduAPI. Each
   lab's managed_env() needs a small change - see fast_managed_api()'s
   docstring for the exact swap.

Uses sphero_unsw (not the upstream spherov2) throughout, and scans for the
toy the same way src/sphero_env/robot/connect.py and
sphero_unsw/toys_scanner.py already do - unfiltered by toy type (so
BOLT/BOLT+ both work), with an interactive pick-from-a-list prompt.

--- Why this exists ---
Every command goes through Toy._execute(), which:
  1. Writes the packet over BLE with `write_gatt_char(..., response=True)`
     (waits for the BLE-layer ack), then
  2. Blocks the caller waiting for the Sphero's own application-level
     response packet, and
  3. Funnels ALL commands through one background thread that sleeps
     `cmd_safe_interval` (75ms for BOLT) after every single write.
That's three stacked sources of delay for what should be a fire-and-forget
drive command in a tight control loop.

--- What this does instead ---
Packet *building* (checksums, framing, sequence numbers) is genuinely fiddly
to get right, so we reuse sphero_unsw's own encoder (`Drive._encode`) rather
than reimplementing the protocol. We only replace how the built bytes get
sent: straight to the underlying bleak BLE characteristic with
`response=False` and no imposed sleep, skipping all three delays above.

This reaches into a few name-mangled private attributes of sphero_unsw's Toy
and BleakAdapter classes (documented at each use site) since the public API
doesn't expose a fire-and-forget send path. That's inherently fragile across
library versions - if this breaks after an upgrade, it's almost always
because one of these private attribute names changed.
"""
import asyncio
import functools
from contextlib import contextmanager
from typing import Callable, Optional

from sphero_unsw import scanner
from sphero_unsw.commands.drive import Drive, DriveFlags
from sphero_unsw.helper import to_bytes
from sphero_unsw.toy import Toy
from sphero_unsw.utils import ToyUtil


class FastSpheroLink:
    """Low-latency drive + trimmed-sensor link to a Sphero toy."""

    def __init__(self, toy: Toy):
        self.toy = toy
        self._latest_sensors: dict = {}

        # SpheroEduAPI.__enter__() normally does toy.wake() +
        # ToyUtil.set_robot_state_on_start(toy) before anything else - a raw
        # Toy.__enter__() (which is all `with toy:` gives you) skips both,
        # so drive commands are silently ignored and the locator never gets
        # reset/enabled. Reusing sphero_unsw's own init here (not
        # SpheroEduAPI - that's the class we're bypassing) instead of
        # hand-rolling a partial version of it.
        toy.wake()
        ToyUtil.set_robot_state_on_start(toy)

        # lab1.py/lab2.py both call api.reset_aim() right after connecting -
        # without it, heading_cmd=0 means "whatever direction the ball's
        # yaw reference happened to be left at" (e.g. from a previous
        # session), not "forward" - so a computed heading_cmd can point the
        # ball in a completely different real-world direction than the math
        # assumes. This recalibrates 0deg to the ball's current physical
        # orientation, same as reset_aim() does via SpheroEduAPI.
        # Physical setup requirement (same as lab1/lab2): orient the ball to
        # face your chosen "+y" direction before connecting.
        toy.reset_yaw()

        # Reach into Toy's private adapter/event loop so we can write bytes
        # directly, bypassing Toy._execute()'s ack-wait and the
        # cmd_safe_interval-throttled background sender thread entirely.
        adapter = getattr(toy, "_Toy__adapter", None)
        if adapter is None:
            raise RuntimeError(
                "Toy has no adapter yet - use FastSpheroLink inside the "
                "toy's `with` block (after __enter__ has run)."
            )
        self._device = getattr(adapter, "_BleakAdapter__device", None)
        self._loop = getattr(adapter, "_BleakAdapter__event_loop", None)
        if self._device is None or self._loop is None:
            raise RuntimeError(
                "Could not reach BleakAdapter internals - spherov2's "
                "adapter implementation may have changed."
            )

        # Plain BOLT drives on Processors.SECONDARY; BOLTPLUS drives on
        # Processors.PRIMARY (see BOLT.drive_with_heading vs
        # BOLTPLUS.drive_with_heading in sphero_unsw/toy/{bolt,boltplus}.py)
        # - using the wrong one gets a well-formed packet the firmware just
        # ignores. Read it off the toy's own class instead of hardcoding
        # either value, so this works for whichever model is connected.
        # NOTE: must use __dict__, not getattr() - accessing a partialmethod
        # through the class via getattr() invokes its descriptor protocol
        # and returns a plain function with no .keywords, silently losing
        # the proc value (that cost us a debugging round-trip already).
        self._drive_proc = None
        for klass in type(toy).__mro__:
            raw = klass.__dict__.get("drive_with_heading")
            if isinstance(raw, functools.partialmethod):
                self._drive_proc = raw.keywords.get("proc")
                break
        if self._drive_proc is None:
            raise RuntimeError(
                f"Could not determine the drive processor target for "
                f"{type(toy).__name__} - check drive_with_heading's proc= "
                f"on this toy class in sphero_unsw."
            )

    # ---- Fast drive path (bypasses spherov2's blocking command path) ---- #

    def _fast_write(self, uuid: str, data: bytes) -> None:
        """Write raw bytes to a BLE characteristic with no response-wait and
        no cmd_safe_interval throttle. Chunked to 20 bytes like spherov2's
        own sender, in case a future command payload exceeds that."""
        while data:
            chunk, data = data[:20], data[20:]
            fut = asyncio.run_coroutine_threadsafe(
                self._device.write_gatt_char(uuid, chunk, False),  # response=False
                self._loop,
            )
            fut.result(timeout=1.0)  # waits for local dispatch only, not a device ack

    def fast_drive(self, heading_deg: int, speed: int) -> None:
        """Send a drive_with_heading command immediately, skipping the ack
        wait and the 75ms enforced gap the stock library imposes."""
        heading_deg = int(heading_deg) % 360
        flags = DriveFlags.FORWARD
        if speed < 0:
            flags = DriveFlags.BACKWARD
            heading_deg = (heading_deg + 180) % 360
        speed = min(255, abs(int(speed)))

        # Must target the same processor this toy's own drive_with_heading
        # uses (BOLT: SECONDARY, BOLTPLUS: PRIMARY) - the wrong one gets a
        # well-formed packet the firmware just ignores, with no error.
        packet = Drive._encode(self.toy, 7, self._drive_proc, [speed, *to_bytes(heading_deg, 2), flags])
        self._fast_write(self.toy._send_uuid, packet.build())

    def fast_stop(self, heading_deg: int = 0) -> None:
        self.fast_drive(heading_deg, 0)

    # ---- Trimmed sensor streaming ---- #

    def configure_sensors(self, sensors=("locator",), interval_ms: int = 33,
                           on_update: Optional[Callable[[dict], None]] = None) -> None:
        """Stream only the sensors actually needed (default: just position).

        Dropping accelerometer/gyroscope/attitude/etc cuts the size of every
        streamed packet and the CPU work spent decoding it - for a PID +
        maze-navigation loop that only needs position (and maybe heading),
        those extra sensors are pure overhead.
        """
        if not hasattr(self.toy, "sensor_control"):
            raise RuntimeError("This toy has no sensor_control (unsupported model?).")
        self.toy.sensor_control.enable(*sensors)
        self.toy.sensor_control.set_interval(int(interval_ms))

        def _listener(data: dict):
            self._latest_sensors.update(data)
            if on_update is not None:
                on_update(self._latest_sensors)

        self.toy.sensor_control.add_sensor_data_listener(_listener)

    def get_location(self):
        """Latest cached {"x":.., "y":..} in cm, or None if no reading yet.

        Same shape as SpheroEduAPI.get_location() - see the
        Robot-compatibility section below for why that matters.
        """
        return self._latest_sensors.get("locator")

    def get_velocity(self):
        """Same shape as SpheroEduAPI.get_velocity(): {"x":.., "y":..} or None."""
        return self._latest_sensors.get("velocity")

    def get_acceleration(self):
        """Same shape as SpheroEduAPI.get_acceleration(); None unless
        "accelerometer" was included in configure_sensors()."""
        return self._latest_sensors.get("accelerometer")

    def get_gyroscope(self):
        """Same shape as SpheroEduAPI.get_gyroscope(); None unless
        "gyroscope" was included in configure_sensors() (it's an extended
        sensor - costs more than the default locator-only set)."""
        return self._latest_sensors.get("gyroscope")

    def get_orientation(self):
        """Same shape as SpheroEduAPI.get_orientation() ({"pitch","roll","yaw"});
        None unless "attitude" was included in configure_sensors()."""
        return self._latest_sensors.get("attitude")

    # ---- SpheroEduAPI-compatible surface, so Robot(api=link, ...) works ---- #
    #
    # Robot (src/sphero_env/robot/robot.py) only ever calls a fixed set of
    # named methods on whatever object is passed as `api`. Implementing that
    # same surface here means Robot can be constructed with a FastSpheroLink
    # in place of a real SpheroEduAPI and drive through the fast path with
    # ZERO changes to Robot, controller.py, EKF.py, Planner.py, or any lab's
    # control loop - only the connection setup in each lab's managed_env()
    # needs to change (see fast_managed_api() below).
    #
    # The one real cost: Robot.set_heading_and_speed() calls api.set_heading()
    # then api.set_speed() as two separate calls, so this sends two fast
    # writes per step instead of one combined fast_drive(). Given the fast
    # path is already vastly cheaper per call than the stock ack-wait, that's
    # a small price for not having to touch Robot itself.

    def reset_aim(self):
        """Same as SpheroEduAPI.reset_aim() - recalibrates 0deg to the ball's
        current physical orientation. Already done once in __init__; calling
        it again (e.g. because existing lab code does `api.reset_aim()`
        right after connecting) is harmless."""
        self.toy.reset_yaw()

    def set_heading(self, heading_deg: int) -> None:
        self._last_heading_deg = int(heading_deg) % 360
        self.fast_drive(self._last_heading_deg, getattr(self, "_last_speed_raw", 0))

    def set_speed(self, speed_raw: int) -> None:
        self._last_speed_raw = int(max(0, min(255, speed_raw)))
        self.fast_drive(getattr(self, "_last_heading_deg", 0), self._last_speed_raw)

    def get_heading(self) -> int:
        """Readback of the last commanded heading, in degrees - same
        contract as SpheroEduAPI.get_heading() (which is also just a
        readback of the last command, not an independent measurement)."""
        return getattr(self, "_last_heading_deg", 0)

    def get_speed(self) -> int:
        """Readback of the last commanded speed, 0-255 - same contract as
        SpheroEduAPI.get_speed()."""
        return getattr(self, "_last_speed_raw", 0)


def connect(scanning_time: float = 3.0) -> Toy:
    """Scan for nearby Sphero toys and let the user pick one - same logic as
    sphero_unsw.toys_scanner.toys_scanner.scan_and_select_toy() /
    src/sphero_env/robot/connect.py's scan_and_connect(), which find the toy
    reliably. Unfiltered by toy type (unlike spherov2.scanner.find_BOLT,
    which failed to match this hardware).

    Returns an unconnected toy - caller opens the actual BLE connection by
    using it as a context manager: `with connect() as toy: ...`.
    """
    while True:
        print(f"\nScanning for Sphero toys for {scanning_time} seconds...")
        toys = scanner.find_toys(timeout=scanning_time)

        if not toys:
            print("No Sphero toys found. Ensure your Bluetooth is on and the toy is awake.")
        else:
            print("\nAvailable Sphero toys:")
            for idx, toy in enumerate(toys, start=1):
                print(f"{idx}. {toy.name}")

        choice = input("\nPress 'ENTER' to rescan, or enter the number of the toy to connect: ").strip()
        if choice == "":
            continue
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(toys):
                selected = toys[choice_num - 1]
                print(f"Selected: {selected.name}\n")
                return selected
            print(f"Please enter a number between 1 and {len(toys)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(toys)} or press ENTER to rescan.")


@contextmanager
def fast_managed_api(sensors=("locator",), interval_ms: int = 33):
    """Drop-in replacement for `with SpheroEduAPI(toy) as api:` that drives
    through the fast path instead - the object it yields implements the same
    methods Robot calls on `api`, so Robot(api=..., ...) works unmodified.

    To switch a lab from SpheroEduAPI to fast_comms, in that lab's
    managed_env() (or equivalent), change:

        selected_toy, _ = scan_and_connect()
        api = stack.enter_context(SpheroEduAPI(selected_toy))
        api.reset_aim()
        real_env = make_real_env(api)

    to:

        import sys; sys.path.insert(0, "<path to>/labs/lab2/fast_comms")
        from fast_link import fast_managed_api
        api = stack.enter_context(fast_managed_api())
        real_env = make_real_env(api)   # <-- unchanged

    `make_real_env`, controller.py, EKF.py, Planner.py and the control loop
    itself don't need to change at all.

    sensors/interval_ms control what Robot's collision sensing has access
    to: the default (locator only) means Robot._sense_collision() always
    reports no collision (get_acceleration()/get_gyroscope() return None) -
    lab1/lab2/lab3's control code doesn't use collision, so this default
    favours speed. Pass e.g. sensors=("locator","accelerometer","velocity",
    "gyroscope") to restore real collision detection at the cost of more
    BLE sensor traffic.
    """
    toy = connect()
    with toy:
        link = FastSpheroLink(toy)
        link.configure_sensors(sensors=sensors, interval_ms=interval_ms)
        yield link
