# STM32MP257 Debian 12 Installation

Target platform:

- STM32MP257 Debian 12
- Python 3.11
- ARM64
- Lingxiao flight controller through direct serial
- D500 radar through direct Linux UART/USB serial
- Headless V4L2 camera and ONNXRuntime CPU inference
- Raw `pyrealsense2` backend for T265

## System Packages

Install Python and native Debian packages first:

```bash
sudo apt update
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  python3-numpy \
  python3-scipy \
  python3-opencv \
  python3-matplotlib \
  libopenblas0 \
  liblapack3 \
  udev
```

OpenCV, Numpy, Scipy, and Matplotlib are installed with `apt` on Debian ARM64.
They are not installed through the main pip requirements.

## Python Environment

Create the project virtual environment:

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The `--system-site-packages` option is required so the virtual environment can use
the Debian `apt` packages for `numpy`, `scipy`, `opencv`, and `matplotlib`.

## T265 Backend

T265 is a required project function, but `pyrealsense2` is a native Intel
RealSense binding. Install and validate it separately so T265 deployment issues
are easy to locate:

```bash
source .venv/bin/activate
pip install -r requirements-t265.txt
python3 -c "import pyrealsense2 as rs; print('pyrealsense2 import OK:', rs)"
```

## Serial Permissions

Add the runtime user to the `dialout` group for serial devices:

```bash
sudo usermod -aG dialout $USER
```

Log out and log back in after changing group membership.

Do not execute `sudo chmod 777` from the program. Device permissions should be
handled with Linux groups or udev rules.

## Basic Validation

Run the basic import checks after installing dependencies:

```bash
source .venv/bin/activate
python3 -m compileall FlightController
python3 -c "from FlightController import FC_Controller; print('FlightController import OK')"
```

Run the full no-hardware environment check after all Python dependencies are
installed. This validates imports, native bindings, and pure software smoke
paths without opening serial ports, cameras, D500 radars, T265, or model files:

```bash
source .venv/bin/activate
python3 tools/check_environment_no_hardware.py
```

## Validation Sequence

Run the no-hardware checks first, then hardware smoke tests with stable device
paths from `/dev/serial/by-id` and `/dev/v4l/by-id`:

```bash
python3 tools/check_environment_no_hardware.py
python3 tools/validate_imports.py
python3 tools/smoke_no_hardware.py
python3 tools/smoke_fc_serial.py --port /dev/serial/by-id/xxx_fc
python3 tools/smoke_dual_radar.py --front-port /dev/serial/by-id/xxx_front --rear-port /dev/serial/by-id/xxx_rear
python3 tools/smoke_t265.py
python3 tools/smoke_camera_ai.py --device /dev/v4l/by-id/xxx_camera
```
