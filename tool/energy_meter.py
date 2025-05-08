#!/usr/bin/env python
"""
This module implements the class EnergyMeter, to measure the energy consumption of Python 
functions or code chunks, segregating their energy usage per component (CPU, DRAM, GPU and 
Hard Disk). 
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyRAPL
import pynvml
import subprocess
import os
import shlex
import json
import time
import threading


class ThreadGpuSamplingPyNvml(threading.Thread):
    """Thread to sample the power draw of the GPU using pynvml.
    """
    SECONDS_BETWEEN_SAMPLES = 0.5

    def __init__(self, name):
        """Init the thread variables and the NVML instance to be queried later on.
        """
        threading.Thread.__init__(self)
        self.name = name
        self.stop = False
        self.power_draw_history = []
        self.activity_history = []
        pynvml.nvmlInit()
        self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU

    def run(self):
        """Start the sampling and stop when self.stop == True.
        """
        while not self.stop:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0  # Convert from milliwatts to watts
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle).gpu
            self.power_draw_history.append(power_draw)
            self.activity_history.append(utilization)
            time.sleep(ThreadGpuSamplingPyNvml.SECONDS_BETWEEN_SAMPLES)

    def stop_sampling(self):
        """Stop the sampling and shutdown NVML.
        """
        self.stop = True
        pynvml.nvmlShutdown()


class EnergyMeter:
    """
    The consumption of each component is measured as follows:

    - CPU: the energy consumption of the CPU is measured with RAPL via the pyRAPL
        library. RAPL is an API from Intel, which is also semi-compatible with
        AMD. RAPL on Intel has been shown to be accurate thanks to the usage of
        embedded sensors in the processor and memory while AMD uses performance
        counters and is therefore, not so accurate.

    - DRAM: the energy used by the memory is also measured with RAPL via pyRAPL.

    - GPU: we measure the energy consumption of the GPU using pynvml in a separate thread.

    - Disk: we estimate the energy consumption of the disk based on read and write activity.
    """
    SCRIPT = (
        "tracepoint:syscalls:sys_enter_write {@wbytes[comm] = sum(args->count);} "
        "tracepoint:syscalls:sys_enter_read {@rbytes[comm] = sum(args->count);}"
    )

    def __init__(self, disk_avg_speed, disk_active_power, disk_idle_power, 
                 label=None, include_idle=False):
        self.label = label if label else "Meter"
        self.include_idle = include_idle

        # Setup pyRAPL to measure CPU and DRAM.
        pyRAPL.setup()

        # Setup disk parameters.
        self.disk_avg_speed = disk_avg_speed
        self.disk_active_power = disk_active_power
        self.disk_idle_power = disk_idle_power

        # Create thread for sampling the power draw of the GPU using pynvml.
        self.thread_gpu = ThreadGpuSamplingPyNvml("GPU Sampling Thread")

        # Create the pyRAPL meter to measure CPU and DRAM energy consumption.
        self.meter = pyRAPL.Measurement(self.label)

        # Create command for bpftrace subprocess that will count the bytes read and
        # written to disk.
        self.bpftrace_command = shlex.split(
            "sudo bpftrace -f json -e '{}'".format(EnergyMeter.SCRIPT)
        )

    def begin(self):
        """Begin measuring the energy consumption."""
        self.meter.begin()
        self.popen = subprocess.Popen(
            self.bpftrace_command, stdout=subprocess.PIPE, preexec_fn=os.setpgrp
        )
        self.bpftrace_pid = os.getpgid(self.popen.pid)
        self.thread_gpu.start()

    def end(self):
        """Finish the measurements and calculate results for CPU, DRAM, and GPU."""
        self.meter.end()
        subprocess.check_output(shlex.split("sudo kill {}".format(self.bpftrace_pid)))
        self.thread_gpu.stop_sampling()
        self.thread_gpu.join()
        po = self.popen.stdout.read()
        self.total_rbytes, self.total_wbytes = self.__preprocess_bpftrace_output(po)

    def __preprocess_bpftrace_output(self, bpftrace_output):
        """Preprocess the output of the bpftrace script."""
        bpftrace_output = bpftrace_output.decode()
        if len(bpftrace_output.strip()) > 0:
            po = bpftrace_output.split("\n")
            rbytes = json.loads(po[3]).get("data").get("@rbytes", {})
            wbytes = json.loads(po[4]).get("data").get("@wbytes", {})
            total_rbytes = rbytes.get("python", 0) + rbytes.get("python3", 0)
            total_wbytes = wbytes.get("python", 0) + wbytes.get("python3", 0)
        else:
            total_rbytes = total_wbytes = 0
        return total_rbytes, total_wbytes

    def get_total_joules_disk(self):
        tot_bytes = self.total_rbytes + self.total_wbytes
        disk_active_time = tot_bytes / self.disk_avg_speed
        disk_idle_time = self.meter.result.duration * 1e-6 - disk_active_time
        te = disk_active_time * self.disk_active_power
        if self.include_idle:
            te += disk_idle_time * self.disk_idle_power
        return te

    def get_total_joules_cpu(self):
        return np.array(self.meter.result.pkg) * 1e-6

    # def get_total_joules_dram(self):
    #     return np.array(self.meter.result.dram) * 1e-6
    
    def get_total_joules_dram(self):
        """We obtain the total joules consumed by the DRAM from pyRAPL."""
        if self.meter.result.dram is None:
            return 0  # 返回 0 或其他默认值
        return np.array(self.meter.result.dram) * 1e-6


    def get_total_joules_gpu(self):
        if len(self.thread_gpu.activity_history) == 0:
            return 0
        if self.include_idle:
            mean_p = np.mean(self.thread_gpu.power_draw_history)
            te = mean_p * self.meter.result.duration * 1e-6
        else:
            if sum(self.thread_gpu.activity_history) == 0:
                return 0
            pdh = self.thread_gpu.power_draw_history
            ah = self.thread_gpu.activity_history
            assert len(pdh) == len(ah), "Power draw and activity history have different lengths!"
            sbs = self.thread_gpu.SECONDS_BETWEEN_SAMPLES
            mean_p = np.mean([pdh[i] for i in range(len(pdh)) if ah[i] > 0])
            te = mean_p * min(self.meter.result.duration * 1e-6, len(ah) * sbs)
        return te

    def get_total_joules_per_component(self):
        return {
            "cpu": self.get_total_joules_cpu(),
            "dram": self.get_total_joules_dram(),
            "gpu": self.get_total_joules_gpu(),
            "disk": self.get_total_joules_disk(),
        }

    def plot_total_joules_per_component(self, include_total=True):
        data = self.get_total_joules_per_component()
        if include_total:
            data["total"] = (
                np.sum(data.get("cpu"))
                + np.sum(data.get("dram"))
                + data.get("disk")
                + data.get("gpu")
            )
        keys = data.keys()
        values = [float(val) for val in data.values()]

        fig, ax = plt.subplots()
        bars = ax.bar(list(keys), values)
        ax.bar_label(bars)
        plt.xlabel("Components")
        plt.ylabel("Joules")
        plt.title(self.meter.label)
        plt.show()
