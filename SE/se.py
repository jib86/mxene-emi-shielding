import pandas as pd
import numpy as np
# Constants
epsilon_0 = 8.85e-12  # Permittivity of free space in F/m
Z_0 = 377             # Vacuum impedance in Ohms
h = 4.135667696e-15   # Planck's constant in eV·s

# Load the data
data = pd.read_csv('resultado.dat', sep='\s+', header=None)
data.columns = ['energy_eV', 'sigma1', 'sigma2']
# Convert columns to numeric types and remove any rows with NaN or zero values
data = data.apply(pd.to_numeric, errors='coerce').dropna()
data = data[data['energy_eV'] != 0]
# Define the thickness
d = 1.26e-9  # Replace this with the actual thickness m
# Transform sigma to sigma_s
data['sigma1'] *= d
data['sigma2'] *= d
# Convert energy from eV to frequency in Hz
data['frequency_Hz'] = data['energy_eV'] / h

omega = 2 * np.pi * data['frequency_Hz']
data['sigma_complex'] = data['sigma1'] + 1j * data['sigma2']
data['epsilon_complex'] = 1 - 1j * data['sigma_complex']/ (omega * epsilon_0)

# Separar parte real e imaginaria de epsilon
data['epsilon1'] = np.real(data['epsilon_complex'])
data['epsilon2'] = np.imag(data['epsilon_complex'])

# Cálculo de n y k
data['n'] = np.sqrt((np.sqrt(data['epsilon1']**2 + data['epsilon2']**2) + data['epsilon1']) / 2)
data['k'] = np.sqrt((np.sqrt(data['epsilon1']**2 + data['epsilon2']**2) - data['epsilon1']) / 2)

# Cálculo de r y t usando la conductividad compleja
data['r'] = (1 - (data['n'] - 1j*data['k']) - Z_0*data['sigma_complex']) / (1 + (data['n'] - 1j*data['k']) + Z_0*data['sigma_complex'])
data['t'] = 2 / (1 + (data['n'] - 1j*data['k']) + Z_0*data['sigma_complex'])

# Calculate the reflectance, transmittance, and absorption
data['R'] = np.power(np.abs(data['r']), 2)
data['T'] = np.real(data['n']) * np.power(np.abs(data['t']), 2)
data['A'] = np.abs(1 - data['R'] - data['T'])

# Cálculo de la impedancia
data['Z'] = Z_0 / np.sqrt(data['epsilon_complex'])

# Separar parte real e imaginaria de la impedancia
data['Z_real'] = np.real(data['Z'])
data['Z_imag'] = np.imag(data['Z'])

# Magnitud de la impedancia
data['Z_mag'] = np.abs(data['Z'])
# Calculate the shielding effectiveness
data['SE_R'] = 10 * np.log10(1 / (1 - data['R'] + 1e-10))
data['SE_A'] = 10 * np.log10((1 - data['R'] + 1e-10) / (data['T'] + 1e-10))
data['SE_T'] = 10 * np.log10(1 / (data['T'] + 1e-10))

SE_R_mean = data['SE_R'].mean()
SE_A_mean = data['SE_A'].mean()
SE_T_mean = data['SE_T'].mean()
Z_mean = data['Z_mag'].mean()
ser = np.power(10, (SE_T_mean /10))                                                                                 
se_percent = (1 - (1 / ser)) * 100
# Handle infinities resulting from division by zero

# Prepare the output data
output_data = data[['frequency_Hz', 'n', 'r', 't', 'R', 'T', 'A', 'SE_R', 'SE_A', 'SE_T', 'Z_mag']]
# Round the numbers to 2 decimal places
output_data = output_data.round(4)
# Set the maximum number of rows to display to None
pd.set_option('display.max_rows', None)
# Convert the output data to a string with columns separated by three spaces
output_string = output_data.to_string(index=False, header=True, col_space=0, justify='left')
# Write the string to a .dat file
with open('output.dat', 'w') as f:
    f.write(output_string)
# Print the output data to the console
print(output_string)
# Print the mean of SE_TOTAL
print(f'The mean of SE_R is: {SE_R_mean} dB')
print(f'The mean of SE_A is: {SE_A_mean} dB')
print(f'The mean of SE_T is: {SE_T_mean} dB')
print(f'The mean of Z is: {Z_mean} Ohms')
print(f'%:{se_percent:}')
