"""Standalone, low-latency Sphero link - bypasses sphero_unsw's per-command
ack-wait and fixed inter-command sleep for drive commands.

This is completely independent of sphero_env/Robot/SpheroEduAPI - it does not
touch any existing lab code. Import and use it directly, or run
benchmark.py to measure the achievable rate on your hardware.

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
from typing import Callable, Optional

from sphero_unsw import scanner
from sphero_unsw.commands.drive import Drive, DriveFlags
from sphero_unsw.controls.v2 import Processors
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

        # BOLT's drive_with_heading targets the SECONDARY processor (the
        # motor MCU) - see BOLT.drive_with_heading in sphero_unsw/toy/bolt.py.
        # Without this, the packet is well-formed and sends fine but the
        # firmware silently ignores it since it's addressed to the wrong
        # processor - that's what looked like a suspiciously-high, entirely
        # fake "844 Hz" with zero actual movement.
        packet = Drive._encode(self.toy, 7, Processors.SECONDARY, [speed, *to_bytes(heading_deg, 2), flags])
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
        """Latest cached (x, y) in cm, or None if no locator reading yet."""
        loc = self._latest_sensors.get("locator")
        if loc is None:
            return None
        return loc.get("x"), loc.get("y")

    def get_velocity(self):
        vel = self._latest_sensors.get("velocity")
        if vel is None:
            return None
        return vel.get("x"), vel.get("y")


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
