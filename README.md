Troubleshooting
pyRAPL/RAPL
pyRAPL requires access to /sys/class/powercap/intel-rapl, for which sudo access is required. If the access is denied, run the following command on the terminal to enable access to the rapl measurement:

sudo chmod -R a+r /sys/class/powercap/intel-rapl

bpftrace
Also note that you need to have bpftrace installed. On ubuntu, you can install it with the following command:

sudo apt-get install -y bpftrace

For other operating systems, please check https://github.com/bpftrace/bpftrace/blob/master/INSTALL.md.