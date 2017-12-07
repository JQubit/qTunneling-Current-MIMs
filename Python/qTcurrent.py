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
from scipy import integrate as sc_int


plt.figure(1)
plt.plot(a2_grid[0, 0, :], prob_quad_a2, '-', color='orange', label='Prob. Marginal')
plt.title('Probabilidad Marginal de $a_2$')
plt.xlabel('$a_2$', fontsize=12)
plt.ylabel('$P_{a_2}^M$', fontsize=12)
plt.axvline(x=mean_a2, ls='--', color='r', label='Media = ' + '%.3E' % Decimal(str(mean_a2)))
plt.axvline(x=mean_a2 + std_a2, ls='-.', color='g')
plt.axvline(x=mean_a2 - std_a2, ls='-.', color='g')
plt.fill_between(a2_grid[0, 0, :], prob_quad_a2, where=credibility_region_1sigma_a2, color=(204/255, 1, 153/255))
plt.legend()
plt.grid(linestyle='--')


# Guardar gráficos como archivos .eps (vectorial) y .png (mapa de bits):
# (Esto es para agregar al informe posteriormente (el .eps) y para ver sin utilizar python cuando se desee (.png))
plt.figure(1)
plt.savefig('P1_plot_datos.eps')
plt.savefig('P1_plot_datos.png')

# Ver gráficos con IPython:
plt.show()
