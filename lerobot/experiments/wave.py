import requests
import argparse
import json
import time
import threading


class WaveRoverControl:
    def __init__(self, ip_address):
        """Initialize WaveRoverControl with the robot's IP address."""
        self.ip_address = ip_address
        self.base_url = f"http://{ip_address}/js"
        # Compensation factors for left and right motors (1.0 means no compensation)
        self.left_compensation = 1.0
        self.right_compensation = 1.0
        # Current speed values
        self.current_left_speed = 0
        self.current_right_speed = 0
        # Watchdog control
        self.watchdog_active = False
        self.watchdog_thread = None
        self.watchdog_stop_event = threading.Event()
        self.first_command_sent = False

    def set_compensation_factors(self, left_factor=1.0, right_factor=1.0):
        """Set compensation factors for left and right motors."""
        self.left_compensation = left_factor
        self.right_compensation = right_factor

    def send_command(self, command):
        """Send a JSON command to the robot and return the response."""
        try:
            url = f"{self.base_url}?json={json.dumps(command)}"
            response = requests.get(url)
            return response.text
        except requests.RequestException as e:
            print(f"Error sending command: {e}")
            return None

    def _watchdog_loop(self):
        """Background thread to resend speed commands every 2 seconds."""
        while not self.watchdog_stop_event.is_set():
            if self.first_command_sent:
                self.set_speed(self.current_left_speed, self.current_right_speed, resend=True)
            time.sleep(2)

    def start_watchdog(self):
        """Start the watchdog thread."""
        if not self.watchdog_active:
            self.watchdog_active = True
            self.watchdog_stop_event.clear()
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop)
            self.watchdog_thread.daemon = True
            self.watchdog_thread.start()

    def stop_watchdog(self):
        """Stop the watchdog thread."""
        if self.watchdog_active:
            self.watchdog_stop_event.set()
            self.watchdog_thread.join()
            self.watchdog_active = False
            self.first_command_sent = False

    def set_speed(self, left_speed, right_speed, resend=False):
        """
        Set the speed of both motors.
        Args:
            left_speed (float): Speed for left motor (-1.0 to 1.0)
            right_speed (float): Speed for right motor (-1.0 to 1.0)
            resend (bool): Whether this is a watchdog resend
        """
        # Store current speeds
        self.current_left_speed = left_speed
        self.current_right_speed = right_speed

        # Apply compensation factors
        left_speed = left_speed * self.left_compensation
        right_speed = right_speed * self.right_compensation

        # Clamp speeds to valid range (-0.5 to 0.5 as per documentation)
        left_speed = max(-0.5, min(0.5, left_speed))
        right_speed = max(-0.5, min(0.5, right_speed))

        command = {
            "T": 1,  # CMD_SPEED_CTRL
            "L": left_speed,
            "R": right_speed
        }
        response = self.send_command(command)

        # Handle watchdog logic
        if not resend:
            if not self.first_command_sent:
                self.first_command_sent = True
                self.start_watchdog()
            elif left_speed == 0 and right_speed == 0:
                self.stop_watchdog()

        return response

    def emergency_stop(self):
        """Immediately stop the robot and disable the watchdog."""
        self.stop_watchdog()
        return self.set_speed(0, 0)

    def get_imu_data(self):
        """Get IMU data from the robot."""
        command = {"T": 126}  # IMU data command
        return self.send_command(command)

# rover = WaveRoverControl("192.168.8.127")
# time.sleep(0.1)
# rover.set_speed(-0.1, -0.1)
# time.sleep(2)
# rover.emergency_stop()
