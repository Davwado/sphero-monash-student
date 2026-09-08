"""Standalone, low-latency Sphero BOLT link - bypasses spherov2/sphero_unsw's
per-command ack-wait and fixed inter-command sleep for drive commands.

This is completely independent of sphero_env/Robot/SpheroEduAPI - it does not
touch any existing lab code. Import and use it directly, or run
benchmark.py to measure the achievable rate on your hardware.

--- Why this exists ---
spherov2 (and sphero_unsw, which is a fork of it) sends every command through
Toy._execute(), which:
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
to get right, so we reuse spherov2's own encoder (`Drive._encode`) rather
than reimplementing the protocol. We only replace how the built bytes get
sent: straight to the underlying bleak BLE characteristic with
`response=False` and no imposed sleep, skipping all three delays above.

This reaches into a few name-mangled private attributes of spherov2's Toy
and BleakAdapter classes (documented at each use site) since the public API
doesn't expose a fire-and-forget send path. That's inherently fragile across
spherov2 versions - if this breaks after a library upgrade, it's almost
always because one of these private attribute names changed.
"""
import asyncio
from typing import Callable, Optional

from spherov2 import scanner
from spherov2.commands.drive import Drive, DriveFlags
from spherov2.helper import to_bytes
from spherov2.toy.bolt import BOLT


class FastSpheroLink:
    """Low-latency drive + trimmed-sensor link to a Sphero BOLT."""

    def __init__(self, toy: BOLT):
        self.toy = toy
        self._latest_sensors: dict = {}

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

        packet = Drive._encode(self.toy, 7, None, [speed, *to_bytes(heading_deg, 2), flags])
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


def connect(toy_name: Optional[str] = None, timeout: float = 5.0) -> BOLT:
    """Scan for a BOLT (raises scanner.ToyNotFoundError if none found).

    Returns an unconnected toy - caller opens the actual BLE connection by
    using it as a context manager: `with connect() as toy: ...`.
    """
    return scanner.find_BOLT(toy_name=toy_name, timeout=timeout)
