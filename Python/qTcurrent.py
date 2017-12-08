# -*- coding: utf-8 -*-
"""
Description of the Program:

qTcurrent.py calculates the quantum tunneling electric current (I) as a function of external bias voltage (V):
this is the I-V curves for a given structure defined from the Transmission probability T(E).

T(E) must be known for this program to work. Either analytical or numerical calculations of T(E) representing the
desired structure can be used.

Author: Javier Carrasco Ávila
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy import integrate
from decimal import Decimal


def fermi_dirac_fn(energy, temperature, chemical_potential):
    """
    Fermi-Dirac distribution function. Use (as inputs) floats with units provided by the 'astropy' module to keep
    track of the correct units of the result.
    :param energy: independent variable, the energy (E) as a float number.
    :param temperature: fixed value of temperature (T) of the system, as a float number.
    :param chemical_potential: chemical potential (mu) that characterizes the system at T, as a float number.
    :return: value (float) of the Fermi-Dirac distribution function evaluated in E with the particular mu.
    """
    return 1. / (np.exp(-(energy - chemical_potential) / ((const.k_B * temperature)).to(u.eV)) + 1)


# Parameters:
temperature_float = 300. * u.K
chemical_potential_float = 5.53 * u.eV  # aprox. by Fermi energy (of Au)
energy_initial_float = 0. * u.eV
energy_final_float = 10. * u.eV
energy_step_float = 0.01 * u.eV
energy_steps_number_float = (energy_final_float - energy_initial_float) / energy_step_float
energy_bias_float = 4. * u.eV

energy_array = np.linspace(energy_initial_float.value, energy_final_float.value + energy_step_float.value,
                           energy_steps_number_float.value) * energy_step_float.unit

# Calculate:
fermiDiracLeft_array = fermi_dirac_fn(energy_array, temperature_float, chemical_potential_float)
fermiDiracRight_array = fermi_dirac_fn(energy_array + energy_bias_float, temperature_float, chemical_potential_float)
fermiDiracDiff_array = fermiDiracLeft_array - fermiDiracRight_array

integral_value_float = integrate.trapz(fermiDiracDiff_array, energy_array)

# Plot:
plt.figure(1)
plt.plot(energy_array, fermiDiracDiff_array, '-', color='orange', label='F-D diff.')
plt.plot(energy_array, fermiDiracLeft_array, '--', color='r', label='F-D Left')
plt.plot(energy_array, fermiDiracRight_array, '--', color='m', label='F-D Right')
plt.title('Fermi-Dirac functions')
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Probability Distribution Density', fontsize=12)
plt.axvline(x=chemical_potential_float.value, ls=':', color='g', label='Chem. Pot. = ' +
                                                                  '%.3E' % Decimal(str(chemical_potential_float.value)))
plt.axvline(x=(chemical_potential_float - energy_bias_float).value,
            ls=':', color='b', label='Chem. Pot. - Bias Energy = ' +
                                      '%.3E' % Decimal(str((chemical_potential_float - energy_bias_float).value)))
#plt.axvline(x=(chemical_potential_float + (const.k_B * temperature_float).to(u.eV)).value, ls='-.', color='g')
#plt.axvline(x=(chemical_potential_float - (const.k_B * temperature_float).to(u.eV)).value, ls='-.', color='g')
plt.fill_between(energy_array.value, fermiDiracDiff_array.value, color=(1, 1, 224/255, 0.3),
                 label='Integral = ' + '%.3E' % Decimal(str(integral_value_float.value)))
plt.legend()
plt.grid(linestyle='--')


# Save figures as .eps (vectorial) and .png (bits map) files:
plt.figure(1)
plt.savefig('fermiDiracDiff.eps')
plt.savefig('fermiDiracDiff.png')

# See figures with IPython:
plt.show()
