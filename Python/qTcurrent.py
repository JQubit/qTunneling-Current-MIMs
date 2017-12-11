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


def fermi_dirac_fn(energy, temperature, chemical_potential):
    """
    Fermi-Dirac distribution function. Use (as inputs) floats with units provided by the 'astropy' module to keep
    track of the correct units of the result.
    :param energy: independent variable, the energy (E) as a float number or numpy array of floats.
    :param temperature: fixed value of temperature (T) of the system, as a float number.
    :param chemical_potential: chemical potential (mu) that characterizes the system at T, as a float number.
    :return: value (float or numpy array of floats) of the Fermi-Dirac distribution function evaluated in E
    with the particular mu.
    """
    return 1. / (np.exp((energy - chemical_potential) / (const.k_B * temperature).to(u.eV)) + 1)


def transmission_prob_rectangular_barrier_no_bias_fn(energy, potential_barrier_height, potential_barrier_width,
                                                     effective_mass):
    """
    Transmission probability (T(E)) through (the simplest case of) a rectangular energy potential barrier without
    external bias voltage applied. This is an analytical result.
    :param energy: independent variable, the energy (E) as a float number or numpy array of floats.
    :param potential_barrier_height: height of the energy potential rectangular barrier (U), as a float.
    :param potential_barrier_width: width (w) of U, as a float.
    :param effective_mass: effective mass of the fermion carrier in the material that makes the barrier, as a float.
    :return: value (float or numpy array of floats) of the T(E) function evaluated in E with the particular U and w.
    """
    w = potential_barrier_width.to(u.nm)
    h_bar = const.hbar.to(u.eV * u.s)
    m_eff = effective_mass.to((u.eV * u.s ** 2) / (u.nm ** 2))
    if type(energy.value) == float:
        k = np.sqrt(2. * m_eff * energy) / h_bar
        kappa = np.sqrt(2. * m_eff * abs(potential_barrier_height - energy)) / h_bar
        if energy == 0.:
            return 0.
        elif energy < potential_barrier_height:
            return 1. / (1. + ((k ** 2. + kappa ** 2.) / (2. * k * kappa)) ** 2. * np.sinh(kappa * w * u.rad) ** 2.)
        elif energy > potential_barrier_height:
            return 1. / (1. + ((k ** 2. - kappa ** 2.) / (2. * k * kappa)) ** 2. * np.sin(kappa * w * u.rad) ** 2.)
        elif energy == potential_barrier_height:
            return 1. / (1. + (m_eff * potential_barrier_height * w ** 2.) / (2. * h_bar ** 2.))
        else:
            # As T(E) must be non negative, a value -1 will be clear indication of an error:
            return -1.
    elif type(energy.value) == np.ndarray:
        t = np.array([])
        for energy_element in energy:
            k = np.sqrt(2. * m_eff * energy_element) / h_bar
            kappa = np.sqrt(2. * m_eff * abs(potential_barrier_height - energy_element)) / h_bar
            if energy_element == 0.:
                t_element = 0.
                t = np.append(t, t_element)
            elif energy_element < potential_barrier_height:
                t_element = 1. / (1. + ((k ** 2. + kappa ** 2.) / (2. * k * kappa)) ** 2.
                                  * np.sinh(kappa * w * u.rad) ** 2.)
                t = np.append(t, t_element)
            elif energy_element > potential_barrier_height:
                t_element = 1. / (1. + ((k ** 2. - kappa ** 2.) / (2. * k * kappa)) ** 2.
                                  * np.sin(kappa * w * u.rad) ** 2.)
                t = np.append(t, t_element)
            elif energy_element == potential_barrier_height:
                t_element = 1. / (1. + (m_eff * potential_barrier_height * w ** 2.) / (2. * h_bar ** 2.))
                t = np.append(t, t_element)
            else:
                # As T(E) must be non negative, a value -1 will be clear indication of an error:
                t_element = -1.
                t = np.append(t, t_element)
        return t
    else:
        raise NameError('Incorrect type of "energy (E)" element as input in T(E) function')

# ---------------------------------------------------------------------------------------------------------------------
# Parameters to calculate the transmission probability:
# ---------------------------------------------------------------------------------------------------------------------
fermi_energy_float = 5.53 * u.eV  # fermi energy of gold (Au)
energy_potential_barrier_height_float = 2. * u.eV
energy_potential_barrier_width_float = 1. * u.nm
barrier_effective_mass_float = 1. * const.m_e
energy_step_transmission_prob_float = 0.001 * u.eV
# If we calculate transmission from left to right, we set the left electrode as reference level of zero energy;
# if we calculate transmission from right to left, we set the right electrode as reference level of zero energy:
energy_incoming_carriers_side_reference_float = 0. * u.eV
energy_steps_number_transmission_prob_float = (fermi_energy_float - energy_incoming_carriers_side_reference_float)\
                                               / energy_step_transmission_prob_float + 1.

energy_transmission_prob_array = np.linspace(energy_incoming_carriers_side_reference_float.value,
                                             fermi_energy_float.value,
                                             energy_steps_number_transmission_prob_float.value)\
                                 * energy_step_transmission_prob_float.unit

# ---------------------------------------------------------------------------------------------------------------------
# Calculate the transmission probability curve:
# ---------------------------------------------------------------------------------------------------------------------
transmission_probability_array = transmission_prob_rectangular_barrier_no_bias_fn(energy_transmission_prob_array,
                                                                                  energy_potential_barrier_height_float,
                                                                                  energy_potential_barrier_width_float,
                                                                                  barrier_effective_mass_float)

# ---------------------------------------------------------------------------------------------------------------------
# Plot the transmission probability curve:
# ---------------------------------------------------------------------------------------------------------------------
plt.figure(1)
plt.plot(energy_transmission_prob_array, transmission_probability_array, '-', color='orange',
         label='$T(E), W = $' + str(energy_potential_barrier_width_float))
plt.title('Transmission Probability - Rectangular Barrier - No Bias')
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Transmission Probability', fontsize=12)
plt.axvline(x=energy_potential_barrier_height_float.value, ls=':',
            color='g', label='$U_{barrier}$ = ' + str(energy_potential_barrier_height_float))
plt.axvline(x=fermi_energy_float.value, ls=':', color='b',
            label='$E_f$ = ' + str(fermi_energy_float))
plt.legend()
plt.grid(linestyle='--')

# ---------------------------------------------------------------------------------------------------------------------
# Parameters to calculate the fermi diff. integral:
# ---------------------------------------------------------------------------------------------------------------------
temperature_float = 300. * u.K
chemical_potential_float = fermi_energy_float  # approx. by Fermi energy (of Au)
energy_bias_float = 4. * u.eV
energy_initial_float = 0. * u.eV
energy_final_float = 10. * u.eV
integration_minimum_range_floats_tuple = (chemical_potential_float - 10. * (const.k_B * temperature_float).to(u.eV)
                                          - energy_bias_float,
                                          chemical_potential_float + 10. * (const.k_B * temperature_float).to(u.eV))
# Reduce integration range to the minimum acceptable (defined above), when the integration limits go beyond the
# minimum integration range limits:
if energy_initial_float < integration_minimum_range_floats_tuple[0]:
    energy_initial_float = integration_minimum_range_floats_tuple[0]
if energy_final_float > integration_minimum_range_floats_tuple[1]:
    energy_final_float = integration_minimum_range_floats_tuple[1]
energy_step_float = 0.01 * u.eV
energy_steps_number_float = (energy_final_float - energy_initial_float) / energy_step_float + 1.

energy_array = np.linspace(energy_initial_float.value, energy_final_float.value,
                           energy_steps_number_float.value) * energy_step_float.unit

# ---------------------------------------------------------------------------------------------------------------------
# Calculate the fermi diff. integral:
# ---------------------------------------------------------------------------------------------------------------------
fermiDiracLeft_array = fermi_dirac_fn(energy_array, temperature_float, chemical_potential_float)
fermiDiracRight_array = fermi_dirac_fn(energy_array + energy_bias_float, temperature_float, chemical_potential_float)
fermiDiracDiff_array = fermiDiracLeft_array - fermiDiracRight_array

# Use trapezoidal method to integrate:
fermi_diff_integral_value_float = integrate.trapz(fermiDiracDiff_array, energy_array)

# ---------------------------------------------------------------------------------------------------------------------
# Plot the Fermi-Dirac curves:
# ---------------------------------------------------------------------------------------------------------------------
plt.figure(2)
plt.plot(energy_array, fermiDiracDiff_array, '-', color='orange', label='$n_{FD}^L(E) - n_{FD}^R(E)$')
plt.plot(energy_array, fermiDiracLeft_array, '--', color='r', label='$n_{FD}^L(E)$')
plt.plot(energy_array, fermiDiracRight_array, '--', color='m', label='$n_{FD}^R(E) \equiv n_{FD}^L(E+U_{bias})$')
plt.title('Fermi-Dirac functions')
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Probability Distribution Density', fontsize=12)
plt.axvline(x=chemical_potential_float.value, ls=':',
            color='g', label='$\mu$ = ' + str(chemical_potential_float))
plt.axvline(x=(chemical_potential_float - energy_bias_float).value,
            ls=':', color='b', label='$\mu - U_{bias}$ = '
                                     + str((chemical_potential_float - energy_bias_float)))
# plt.axvline(x=(chemical_potential_float + (const.k_B * temperature_float).to(u.eV)).value, ls='-.', color='g')
# plt.axvline(x=(chemical_potential_float - (const.k_B * temperature_float).to(u.eV)).value, ls='-.', color='g')
plt.fill_between(energy_array.value, fermiDiracDiff_array.value, color=(1, 1, 224/255, 0.3),
                 label='Integral = ' + str(round(fermi_diff_integral_value_float.value, 3)))
plt.legend()
plt.grid(linestyle='--')

# ---------------------------------------------------------------------------------------------------------------------
# Save figures as .eps (vector) and .png (bits map) files:
# ---------------------------------------------------------------------------------------------------------------------
plt.figure(1)
plt.savefig('transmissionProb.eps')
plt.savefig('transmissionProb.png')

plt.figure(2)
plt.savefig('fermiDiracDiff.eps')
plt.savefig('fermiDiracDiff.png')

# ---------------------------------------------------------------------------------------------------------------------
# See figures with IPython:
plt.show()
# ---------------------------------------------------------------------------------------------------------------------
