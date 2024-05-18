from PyMacroFin.model import macro_model
import numpy as np
import pandas as pd
import time
import PyMacroFin.utilities as util
from PyMacroFin.system import system

# initial guess function for endogenous variables
def init_fcn(e,c):
        if e<.3:
                q = 1.05+.06/.3*e
                psi = 1/.3*e
                sigq = -.1*(e-.3)**2+.008
        else:
                psi = 1
                sigq = 0
                q = 1.1 - .03/.7*e
        return [q,psi]

# boundary condition function for eta == 0
def eta_minimum(d):
        psi = 0
        q = (2*d['ah']*d['kappa']+(d['kappa']*d['r'])**2.+1)**0.5 - d['kappa']*d['r']
        return [q,psi]


def define_model(npoints):
        m = macro_model(name='BruSan14_log_utility')

        m.set_endog(['q','psi'],init=[1.05,0.5])
        m.prices = ['q']
        m.set_state(['e'])

        m.params.add_parameter('sig',.1)
        m.params.add_parameter('deltae',.05)
        m.params.add_parameter('deltah',.05)
        m.params.add_parameter('rho',.06)
        m.params.add_parameter('r',.05)
        m.params.add_parameter('ae',.11)
        m.params.add_parameter('ah',.07)
        m.params.add_parameter('kappa',2)

        m.equation('iota = (q**2-1)/(2*kappa)')
        m.equation('phi = 1/kappa*((1+2*kappa*iota)**0.5-1)')
        m.equation('sigq = (((ae-ah)/q+deltah-deltae)/(psi/e-(1-psi)/(1-e)))**0.5 - sig',plot=True,latex=r'$\sigma^q$')
        m.equation('sige = (psi-e)/e*(sig+sigq)')
        m.equation('mue = sige**2 + (ae-iota)/q + (1-psi)*(deltah-deltae)-rho')
        m.equation('er = psi/e*(sig+sigq)**2',plot=True,latex=r'$E[dr_t^k-dr_t]/dt$')
        m.equation('sigee = sige*e',plot=True,latex=r'$\sigma^{\eta} \eta$')
        m.equation('muee = mue*e',plot=True,latex=r'$\mu^{\eta} \eta$')

        m.endog_equation('q*(r*(1-e)+rho*e) - psi*ae - (1-psi)*ah + iota')
        m.endog_equation('(psi-e)*d(q,e) - q*(1-sig/(sig+sigq))')

        m.hjb_equation('mu','e','mue')
        m.hjb_equation('sig','e','sige')

        m.constraint('psi','<=',1,label='upper_psi')
        m.constraint('psi','>=',0,label='lower_psi')

        m.boundary_condition({'e':'min'},eta_minimum)

        s = system(['upper_psi'],m)
        s.equation('sigq = sig/(1-(psi-e)*d(q,e)/q) - sig')
        s.endog_equation('1 - psi')
        s.endog_equation('q*(r*(1-e)+rho*e) - ae + iota')

        m.systems.append(s)

        m.options.ignore_HJB_loop = True
        m.options.import_guess = False
        m.options.guess_function = init_fcn
        m.options.inner_plot = False
        m.options.outer_plot = False
        m.options.final_plot = True
        m.options.n0 = npoints
        m.options.start0 = 0.0
        m.options.end0 = 0.95
        m.options.inner_solver = 'least_squares'
        m.options.derivative_plotting = [('q','e')]
        m.options.min_iter_outer_static = 5
        m.options.min_iter_inner_static = 0
        m.options.max_iter_outer_static = 50
        m.options.return_solution = True
        m.options.save_solution = False
        m.options.price_derivative_method = 'backward'

        return m

if __name__=='__main__':
        npoints = 100
        tic = time.time()
        m = define_model(npoints)
        df = m.run()
        toc = time.time()
        print('elapsed time: {}'.format(toc-tic))