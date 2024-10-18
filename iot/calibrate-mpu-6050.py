import time
import smbus
import json

# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F

bus = smbus.SMBus(1)  # I2C bus number on Raspberry Pi

def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    if value > 32768:
        value = value - 65536
    return value

def initialize_mpu(accel_range):
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00)
    if accel_range == 2:
        accel_config_value = 0x00  # ±2g
        scaling_factor = 16384.0
    elif accel_range == 4:
        accel_config_value = 0x08  # ±4g
        scaling_factor = 8192.0
    elif accel_range == 8:
        accel_config_value = 0x10  # ±8g
        scaling_factor = 4096.0
    elif accel_range == 16:
        accel_config_value = 0x18  # ±16g
        scaling_factor = 2048.0
    else:
        raise ValueError("Invalid accelerometer range.")
    bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, accel_config_value)
    return scaling_factor

def calibrate_sensor(samples=1000, scaling_factor=16384.0):
    acc_x_offset, acc_y_offset, acc_z_offset = 0, 0, 0
    print(f"Starting calibration with {samples} samples...")
    for _ in range(samples):
        acc_x = read_raw_data(ACCEL_XOUT_H)
        acc_y = read_raw_data(ACCEL_YOUT_H)
        acc_z = read_raw_data(ACCEL_ZOUT_H)

        acc_x_offset += acc_x
        acc_y_offset += acc_y
        acc_z_offset += acc_z

        time.sleep(0.008)  # 100Hz

    acc_x_offset /= samples
    acc_y_offset /= samples
    acc_z_offset /= samples

    # Adjust the Z-axis for gravity with respect to the orientation
    # When flat, the raw value should be around ± scaling_factor based on orientation
    if acc_z_offset > 0:
        acc_z_offset -= scaling_factor * 2  # If Z-axis reads positive, subtract 1g (scaling factor)
    else:
        acc_z_offset += scaling_factor * 2  # If Z-axis reads negative, add 1g (scaling factor)


    print(f"Calibration complete.\nOffsets:\nAccX: {acc_x_offset}\nAccY: {acc_y_offset}\nAccZ: {acc_z_offset}")
    return acc_x_offset, acc_y_offset, acc_z_offset

def save_offsets(filename, acc_x_offset, acc_y_offset, acc_z_offset):
    offsets = {
        "acc_x_offset": acc_x_offset,
        "acc_y_offset": acc_y_offset,
        "acc_z_offset": acc_z_offset
    }

    with open(filename, 'w') as f:
        json.dump(offsets, f)
    print(f"Offsets saved to {filename}")

# Main calibration process
accel_range = 16
scaling_factor = initialize_mpu(accel_range)
acc_x_offset, acc_y_offset, acc_z_offset = calibrate_sensor(1000, scaling_factor)
save_offsets("mpu_offsets.json", acc_x_offset, acc_y_offset, acc_z_offset)