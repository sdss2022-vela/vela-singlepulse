
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool,cpu_count
from functools import partial

# BEGIN BEN PERERA CODE

#Get the single pulse (SP) data file list. Note that the
#frequeny channels and two polarization channels have been 
#added in these files. So you have only time resolution. 
fn = glob.glob('sspsr*.Fp')

print('\nList of files:', fn)

#read in the SP data

assert len(fn) != 0

for i in range(len(fn)):
    obs = psrchive.Archive_load(fn[i])
    obs.dedisperse()
    obs.remove_baseline()
    data = obs.get_data()
    #print("data shape",data.shape)
    #print(data)
    #sys.exit()
    if i == 0:
        stack_all = data[:,0,0,:]
    else:
        stack_all = np.concatenate((stack_all,data[:,0,0,:]),axis=0)

#shape of the numpy 2D array
print('\npulse stack array shape:', stack_all.shape)

npulse = stack_all.shape[0]
nbins = stack_all.shape[1]

print('Number of single pulses:', npulse)
print('Number of bins across a pulse:', nbins)


#Averaged pulse profile. Note we center the peak of the pulse across
#the pulse phase
prof = np.mean(stack_all, axis=0)
shift_bins = int(nbins/2 - np.argmax(prof))
stack_all = np.roll(stack_all, shift_bins, axis=1)
prof = np.roll(prof, shift_bins, axis=0)


#plot averaged pulse profile
plt.figure(0)
plt.xlim(0,nbins)
plt.xlabel('Pulse bin')
plt.plot(prof)
plt.title("Average Vela Pulse Profile")
plt.savefig("./data/avg-pulse-profile.png",dpi=600)

#plot individual pulse stack
plt.figure(1)
plt.xlabel('Pulse bin')
plt.ylabel('Pulse number')
plt.imshow(stack_all,aspect='auto',origin='lower',extent=[0,nbins,0,npulse],vmin=-.2,vmax=.2)
plt.title("Vela Individual Pulse Stack")
plt.colorbar()
plt.savefig("./data/individual-pulse-stack.png",dpi=600)

# END BEN PERERA CODE

# Everything below are tasks Ben assigned us

# 1. Now calculate the signal-to-noise ration of each pulse.
#    S/N = peak_amplitude/offpulse_standard_deviation.
# 2. Write a for loop to go through all pulses and get the S/N.
# 3. Write the calculated S/N to a variable.
# 4. What is the highest S/N?
# 5. Can you plot the three highest S/N pulses?
# 6. Do they vary in shape? How can you compare those with the
#    averaged profile?

# DCW: Below is my intial code. It can be parallelized, so I switched to that.
#
# pulse_std = np.zeros(npulse)
# pulse_ampl = np.zeros(npulse)
# signal_indices = np.indices(nbins).squeeze()
#
#
# for i in range(npulse):
#     peak_index = np.argmax(stack_all[i,:])
#     num_cut_indices = 20
#     min_cut = peak_index - num_cut_indices
#     max_cut = peak_index + num_cut_indices
#     signal_slice = stack_all[i][(signal_indices < min_cut) | (signal_indices > max_cut)] # boolean masking. cuts out pulse
#
#     # another way to do this below
#     # signal_slice = np.concatenate((stack_all[i,:min_cut],stack_all[i,max_cut:]))
#     pulse_std[i] = np.std(signal_slice)
#     pulse_ampl[i] = stack_all[i,peak_index]
#
# pulse_s2n = pulse_ampl / pulse_std

def do_pulse_s2n(pulse_number: int, all_pulse_array: np.ndarray, cut_bins:int, avg_peak_index:int ) -> float:
    """Find the signal to noise of a single pulsar pulse.

    Given the array of all pulses, the pulse number you wish to investigate, and
    the number of bins to mask on the right and left side of the pulse peak,
    find the standard deviation and maximum amplitude of the pulse. Use these values
    to find the S/N of the pulse as amplitude/std.

    Parameters
    ----------
    pulse_number : int
        the pulse you wish to calculate statistics for
    all_pulse_array : np.ndarray
        an array consisting of all pulses of shape (npulses, nbins)
    cut_bins : int
        the +- number of bins around the peak to mask out of the pulse data for
        std calculation
    avg_peak_index : int
        the location of the peak of the average pulse

    Returns
    -------
    float
        the S/N

    """

    # Set the region in which to fit the max peak
    min_peak_cut = avg_peak_index - cut_bins // 2
    max_peak_cut = avg_peak_index + cut_bins // 2

    # find the index of the maximum signal in a region around the average pulse peak
    peak_index = np.argmax(all_pulse_array[pulse_number, min_peak_cut:max_peak_cut])

    # Set the regions to exclude
    min_cut = peak_index - cut_bins
    max_cut = peak_index + cut_bins


    # get an array of the indices
    signal_indices = np.indices(all_pulse_array[pulse_number].shape).squeeze() 

    peak_index = np.argmax(all_pulse_array[pulse_number, :])

    # Slice the pulse out of the signal
    # This uses boolean masking. Basically, it just makes an array of True/False values and
    # uses that to pick out only the bins we want.
    signal_slice = all_pulse_array[pulse_number][(signal_indices < min_cut) | (signal_indices > max_cut)]

    # another way to do this
    # signal_slice = np.concatenate((stack_all[i,:min_cut],stack_all[i,max_cut:]))

    off_pulse_mean = np.mean(signal_slice)

    # Subtract off the mean of the off pulse to correct for baseline issues
    all_pulse_array[pulse_number] -= off_pulse_mean

    # Find standard deviation of off pulse
    off_pulse_std = np.std(signal_slice-off_pulse_mean)


    # Find the amplitude of the peak
    pulse_ampl = all_pulse_array[pulse_number,peak_index]

    # Calculate the signal to noise
    s2n = pulse_ampl / off_pulse_std

    return(s2n)


# Set number of indices to use on each side of peak and find average pulse peak location
num_cut_indices = 20

# Find the index of the peak of the average pulse profile
avg_peak_index = np.argmax(prof)

# Do the multiprocessing. This uses functools.partial to return a partial function that has three of its four arguments
# already set. It then is parallelized over the individual pulses.
with Pool(cpu_count() - 2) as pool:
    pulse_s2n = pool.map(partial(do_pulse_s2n, all_pulse_array=stack_all, cut_bins=num_cut_indices,
                                 avg_peak_index=avg_peak_index ), range(npulse))

# convert to numpy array
pulse_s2n = np.array(pulse_s2n)

# Find max S/N
max_pulse_s2n = np.max(pulse_s2n)

# Find the top three S/N
third_s2n , second_s2n, first_s2n = np.sort(pulse_s2n)[-3:]

# check that the first and max agree
assert max_pulse_s2n == first_s2n

# Find where these S/N values occur
first_s2n_index = np.where(pulse_s2n == first_s2n)[0]
second_s2n_index = np.where(pulse_s2n == second_s2n)[0]
third_s2n_index = np.where(pulse_s2n == third_s2n)[0]


# find S/N for average pulse using same method from above
# Do all of the above for the average pulse. Since it is a single pulse
# (not part of a pulse stack), it doesn't work with the function we wrote.
# Do it manually.
min_cut = avg_peak_index - num_cut_indices
max_cut = avg_peak_index + num_cut_indices

avg_signal_slice = np.concatenate((prof[:min_cut],prof[max_cut:]))
prof_std = np.std(avg_signal_slice)
prof_ampl = prof[avg_peak_index]
prof_s2n = prof_ampl / prof_std

# Plots
plt.figure(2)
plt.xlim(0,nbins)
plt.xlabel('Pulse bin')
plt.plot(np.squeeze(stack_all[first_s2n_index])/np.squeeze(np.max(stack_all[first_s2n_index]))
        ,label=f"S/N #1 {float(pulse_s2n[first_s2n_index]):.2f}")

plt.plot(np.squeeze(stack_all[second_s2n_index])/np.squeeze(np.max(stack_all[second_s2n_index]))
        ,label=f"S/N #2 {float(pulse_s2n[second_s2n_index]):.2f}")

plt.plot(np.squeeze(stack_all[third_s2n_index])/np.squeeze(np.max(stack_all[third_s2n_index]))
        ,label=f"S/N #3 {float(pulse_s2n[third_s2n_index]):.2f}")

plt.plot(prof/np.max(prof), label=f"Avg pulse S/N {float(prof_s2n):.2f}")

plt.legend(loc="best")
plt.title("Three Highest S/N Vela Pulses (Normalized)")
plt.savefig("./data/three-highest-sn-pulses.png",dpi=600)



# BEN: Plot the S/N distribution using pyplot hist.

plt.figure(3)
plt.hist(pulse_s2n,bins=nbins)
plt.xlabel("S/N value")
plt.ylabel("Number of pulses")
plt.title("Vela Single Pulse S/N")
plt.savefig("./data/sn-histogram.png",dpi=600)


# BEN: Perhaps you need to using pulses ONLY with S/N > 4.
s2n_limit = 4
filtered_indices = np.where(pulse_s2n > s2n_limit) # pool.map preserves order, so this still works

# BEN: Calculate the on-pulse energy of these selected pulses.
# BEN: Energy is the area under the curve.

# Divide the original num_cut_bins by this for integration region
reduce_by = 4

# Find the energies on and off-pulse
energy_stack = np.trapz(stack_all[filtered_indices[0],
                                  avg_peak_index-num_cut_indices//reduce_by:avg_peak_index+num_cut_indices//reduce_by], axis=1)

energy_stack_off_pulse = np.trapz(stack_all[filtered_indices[0], 0:num_cut_indices//(reduce_by//2)], axis=1)

# Remove extra dimensions
energy_stack_off_pulse.squeeze()
energy_stack.squeeze()

# Normalize with the mean value
energy_stack_normalized = energy_stack / np.mean(energy_stack)
energy_stack_off_pulse_normalized = energy_stack_off_pulse / np.mean(energy_stack)

print(f"Number of pulses with S/N greater than {s2n_limit} is {filtered_indices[0].shape}")
print(f"Average Vela energy {np.mean(energy_stack)}")

# Plot the on and off-pulse energies
plt.figure(4)
plt.hist(energy_stack_normalized, bins=30,label="On-pulse energy",alpha=0.2)
plt.hist(energy_stack_off_pulse_normalized, bins=30, label="Off-pulse energy",alpha=0.2)
plt.xlabel("Energy / average energy")
plt.ylabel("Number of pulses")
plt.legend(loc="best")
plt.title("Vela Single Pulse Energies")
plt.savefig("./data/energy-histogram.png",dpi=600)

# Find equivalent width of the selected pulses
equiv_width = energy_stack \
    / np.max(stack_all[filtered_indices[0],
      avg_peak_index-num_cut_indices//reduce_by:avg_peak_index+num_cut_indices//reduce_by],
      axis=1)

# Convert the bins to ms
period_vela = 89.33 # ms
equiv_width /= nbins # 256 bins
equiv_width *= period_vela

# Plot the on and off-pulse energies
plt.figure(5)
plt.scatter(pulse_s2n[filtered_indices[0]], equiv_width )
plt.xlabel(f"S/N of pulses with S/N > {s2n_limit}")
plt.ylabel("Equivalent width [ms]")
plt.ylim(0,int(equiv_width.max()+1))
plt.title("Vela Equivalent width vs S/N")
plt.savefig("./data/eq-width-vs-sn.png",dpi=600)
