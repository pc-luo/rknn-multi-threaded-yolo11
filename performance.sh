# 请切换到root用户

# CPU定频
echo "CPU可用频率:"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
sudo echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
sudo echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "CPU当前频率:"
sudo watch -n 0.1 cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq