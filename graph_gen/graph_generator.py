import pandas as pd
import matplotlib.pyplot as plt 

times_pp = pd.read_csv('/home/malikali/data_pp/times_pp.csv')
velocities_pp = pd.read_csv('/home/malikali/data_pp/velocities_pp.csv')

times_pid = pd.read_csv('/home/malikali/data_pid/times_pid.csv')
velocities_pid = pd.read_csv('/home/malikali/data_pid/velocities_pid.csv')

time_pp_values = times_pp['Time'].values
velocity_pp_values = velocities_pp['Velocity'].values

time_pid_values = times_pid['Time'].values
velocity_pid_values = velocities_pid['Velocity'].values

plt.figure(figsize=(10,6))
plt.plot(time_pp_values, velocity_pp_values, label='Pure Pursuit')
plt.plot(time_pid_values, velocity_pid_values, label='PID Control')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.grid(True)
plt.show()