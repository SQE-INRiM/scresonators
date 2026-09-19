import numpy as np
import lmfit

from ..utils import *
from .fit_method import FitMethod
from ..utils import find_circle
from scipy.ndimage import gaussian_filter

class LAMBDA2(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(f, Q, f0, Qc1, Qc2, phi1, phi2, n):
        """DCM fitting for all S-parameters altogether"""
        y = np.zeros((len(f), 2, 2), dtype=complex)
        y[:,0,0] = 1-(2*Q*np.exp(1j*phi1)/Qc1)/(1+2j*Q*(f-f0)/f0)
        y[:,1,0] = np.exp(1j*np.pi/2*n) * (2 * Q  *np.exp(1j*(phi1+phi2)/2) / np.sqrt(Qc1*Qc2)) / (1+2j*Q*(f-f0)/f0)
        y[:,0,1] = np.exp(1j*np.pi/2*n) * (2 * Q  *np.exp(1j*(phi1+phi2)/2) / np.sqrt(Qc1*Qc2)) / (1+2j*Q*(f-f0)/f0)
        y[:,1,1] = 1-(2*Q*np.exp(1j*phi2)/Qc2)/(1+2j*Q*(f-f0)/f0)

        penalty = 3*(1-np.cos(np.pi*n)**2)
        return y + penalty

    @staticmethod
    def fit_function(f, params):
        '''
        same as func but unpacks the fit parameters for you.
        '''
        Q = params['Q'].value
        f0 = params['f0'].value
        Qc1 = params['Qc1'].value
        phi1 = params['phi1'].value
        Qc2 = params['Qc2'].value
        phi2 = params['phi2'].value
        n = params['n'].value

        y = np.zeros((len(f), 2, 2), dtype=complex)
        y[:,0,0] = 1-(2*Q*np.exp(1j*phi1)/Qc1)/(1+2j*Q*(f-f0)/f0)
        y[:,1,0] = np.exp(1j*np.pi/2*n) * (2 * Q  *np.exp(1j*(phi1+phi2)/2) / np.sqrt(Qc1*Qc2)) / (1+2j*Q*(f-f0)/f0)
        y[:,0,1] = np.exp(1j*np.pi/2*n) * (2 * Q  *np.exp(1j*(phi1+phi2)/2) / np.sqrt(Qc1*Qc2)) / (1+2j*Q*(f-f0)/f0)
        y[:,1,1] = 1-(2*Q*np.exp(1j*phi2)/Qc2)/(1+2j*Q*(f-f0)/f0)

        penalty = 3*(1-np.cos(np.pi*n)**2)
        return y + penalty

    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['f'])
        return model
    
    def simple_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray):
        f_c = fdata[np.argmin(np.abs(sdata))]

        params = lmfit.Parameters()
        params.add('Q', value=1e6)
        params.add('f0', value=f_c, min=f_c * 0.9, max=f_c * 1.1)
        params.add('Qc1', value=5e5)
        params.add('phi1', value=0, min=-np.pi, max=np.pi)
        params.add('Qc2', value=5e5)
        params.add('phi2', value=0, min=-np.pi, max=np.pi)
        params.add('n', value=0, min=-0.5, max = 3.5)

        return params


    def find_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))#the circle diameter is 4*r = Q/Qc
        """
        Finds the initial guess of the fitting parameters.
        
        In order to robustly estimate the linewidth we smooth the data to eliminate noise.
        Then we calculate |dS/df| to find the region in frequency space where the resonator response is changing rapidly.
        taking the derivative eliminates the need to consider phi for estimating the linewidth.
        Once we have |dS/df| we count up the segments in the frequency data where |dS/df| > cutoff
        This is implemented with the dot product, and that length in frequency space is estimated to be the linewidth.
        The linewidth & f0 determine Q, and the circle diameter is Q/Qc.
        
        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            
        Returns:
            params: lmfit.Parameters object which stores the initial guesses of the fitting parameters
        """
        filtered_data = gaussian_filter(sdata, sigma = 3, axes=0)#sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata, axis=0)

        chiFunction = partitionFrequencyBand(fdata, gradS)

        # chiFunction = np.zeros(len(gradSmagnitude))
        # cutoff = 0.5*(np.min(gradSmagnitude)+np.max(gradSmagnitude))
        # for n in range(len(gradSmagnitude)):
        #     if gradSmagnitude[n] > cutoff:
        #         chiFunction[n] = 1#set to one if |dS/df| is above the cutoff at this point
        linewidth = np.dot(chiFunction[:-1, ...].transpose(1,2,0), np.diff(fdata))


        gradSmagnitude = np.abs(gradS)
        f_c = fdata[np.argmax(gradSmagnitude, axis=0)+3]#uncertainties can't be calculated when this guess is too good!!!
        Q_guess = 2*f_c/(linewidth)
        Qc_guess = Q_guess/(4*r)
        # Qci_guess = Qci + QcT^2 / Qcj, where Qci come from Sii, and QcT = sqrt(QC1*QC2) is the average of "Qc" estimate from S12 and S21
        Qc1_guess = (Qc_guess[0,0] + ((Qc_guess[0,1] + Qc_guess[1,0])/2)**2 / Qc_guess[1,1])/2
        Qc2_guess = (Qc_guess[1,1] + ((Qc_guess[0,1] + Qc_guess[1,0])/2)**2 / Qc_guess[0,0])/2
        # print(r)


        # Create an lmfit.Parameters object to store initial guesses
        params = lmfit.Parameters()
        
        params.add('Q', value=np.mean(Q_guess))
        params.add('f0', value=np.mean(f_c), min=np.min(f_c)*0.9, max=np.max(f_c)*1.1)
        params.add('Qc1', value=Qc1_guess)
        params.add('Qc2', value=Qc2_guess)
        params.add('phi1', value=0, min=-np.pi, max=np.pi)
        params.add('phi2', value=0, min=-np.pi, max=np.pi)
        params.add('n', value=0, min=-0.5, max = 3.5)

        return params

    def extractQi(self, params):
        """
        Calculates the fit value & stddev of Qi.
        """
        Q = params['Q'].value
        Qc = params['Qc'].value
        phi = params['phi'].value
        params.add('inverseQi', value =1/Q - np.cos(phi)/Qc)
        inverseQi = params['inverseQi'].value
        params.add('Qi', value = 1/inverseQi)
        #if you get an error that points here check that all your parameters varied during the fit, set verbose = True
        params['inverseQi'].stderr = np.sqrt((params['Q'].stderr/Q**2)**2+(np.sin(phi)*params['phi'].stderr/Qc)**2+(np.cos(phi)*params['Qc'].stderr/Qc**2)**2)
        params['Qi'].stderr = params['inverseQi'].stderr/inverseQi**2
        return params

    #TODO: add function that converts standard errors to 95% confidence intervals