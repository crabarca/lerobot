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


def main():
    parser = argparse.ArgumentParser(description='Wave Rover Control Script')
    parser.add_argument('ip', type=str, help='IP address of the Wave Rover (e.g., 192.168.10.104)')
    args = parser.parse_args()

    rover = WaveRoverControl(args.ip)

    try:
        while True:
            print("\nWave Rover Control Menu:")
            print("1. Set speed")
            print("2. Get IMU data")
            print("3. Set compensation factors")
            print("4. Emergency stop")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ")

            if choice == "1":
                try:
                    left = float(input("Enter left speed (-1.0 to 1.0): "))
                    right = float(input("Enter right speed (-1.0 to 1.0): "))
                    response = rover.set_speed(left, right)
                    print(f"Response: {response}")
                except ValueError:
                    print("Invalid input. Please enter numbers.")

            elif choice == "2":
                response = rover.get_imu_data()
                print(f"IMU Data: {response}")

            elif choice == "3":
                try:
                    left_comp = float(input("Enter left compensation factor (default 1.0): ") or "1.0")
                    right_comp = float(input("Enter right compensation factor (default 1.0): ") or "1.0")
                    rover.set_compensation_factors(left_comp, right_comp)
                    print("Compensation factors updated.")
                except ValueError:
                    print("Invalid input. Please enter numbers.")

            elif choice == "4":
                response = rover.emergency_stop()
                print("Emergency stop activated.")
                print(f"Response: {response}")

            elif choice == "5":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Emergency stop before exiting
        rover.emergency_stop()


if __name__ == "__main__":
    main()