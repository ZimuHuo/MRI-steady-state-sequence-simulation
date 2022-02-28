'''
SSFP simualtions code 
Author: Zimu Huo
Date: 02.2022
'''

from matplotlib import pyplot as plt
import numpy as np

#SNR -> T2/T1 
def SSFP_plot(M0, alpha, phi, dphi, beta, TR, TE, T1, T2, Nr):
    signal = np.zeros([Nr,3])
    signal[0,:] = [0, 0, M0]
    signal[0,:] = np.matmul(signal[0,:],rotMat(alpha, phi)) 
    for i in range(1, Nr):
        signal[i,0] = signal[i-1,0]*np.exp(-TR/T2)
        signal[i,1] = signal[i-1,1]*np.exp(-TR/T2)
        signal[i,2] = M0+(signal[i-1,2]-M0)*np.exp(-TR/T1)
        P = np.array([
            [ np.cos(beta),  np.sin(beta),   0],
            [-np.sin(beta),  np.cos(beta),   0],
            [            0,             0,   1]
            ])
        signal[i,:] = np.matmul(signal[i,:],rotMat(alpha, phi))
        signal[i,:] = np.matmul(signal[i,:],P)
        phi = phi + dphi
        magnitude = np.sqrt(signal[:,0]*signal[:,0]+signal[:,1]*signal[:,1])
    return magnitude

     
def iterative_SSFP(M0, alpha, phi, dphi, beta, TR, TE, T1, T2, Nr):
    
    '''
    -------------------------------------------------------------------------
    Parameters
    
    M0: scalar  
    Initial magnetization in the B0 field, function of the proton density rho  
    
    Alpha: radian 
    Flip or tip angle of the magnetization vector 
    
    phi: radian 
    Angle between the vector and x axis/ phase
    
    dphi: radian 
    Increment of phi
    
    TR: scalar in msec  
    Repition time of the pulse sequence 
    
    TE: scalar in msec  
    Echo time 
    
    T1: scalar in msec
    T1 value of the tissue
    
    T2: scalar in msec
    T2 value of the tissue
    
    Nr: scalar 
    Number of simulation   
    
    -------------------------------------------------------------------------
    Returns
    Signal : array like
    Signal with x y z value in 1 by 3 vector [x, y, z]
    
    -------------------------------------------------------------------------
    Notes
    Michael's paper section 4 in the end    
    
    -------------------------------------------------------------------------
    References
    
    [1] 
    Author: Michael A. Mendoza
    Title: Water Fat Separation with Multiple Acquisition Balanced Steady State Free Precession MRI
    Link: https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=5303&context=etd
    '''
    
    assert M0 != 0, 'Only non zero numbers are allowed'
    assert alpha != 0, 'Only non zero numbers are allowed'
    assert T1 != 0, 'Only non zero numbers are allowed'
    assert T2 != 0, 'Only non zero numbers are allowed'
    assert TE != 0, 'Only non zero numbers are allowed'
    assert TR != 0, 'Only non zero numbers are allowed'
    assert TE <= TR, 'TE must be shorter than or equal to TR'
    
    
    signal = np.zeros([1,3])
    signal = np.asarray([0, 0, M0])
    
    for i in range(Nr):
        signal = np.matmul(rotMat(alpha, phi),signal) 
        signal[0] = signal[0]*np.exp(-TR/T2)
        signal[1] = signal[1]*np.exp(-TR/T2)
        signal[2] = M0+(signal[2]-M0)*np.exp(-TR/T1)
        P = np.array([
            [ np.cos(beta),  np.sin(beta),   0],
            [-np.sin(beta),  np.cos(beta),   0],
            [            0,             0,   1]
            ])
        signal = np.matmul(P,signal)
        phi = phi + dphi
    signal = np.matmul(rotMat(alpha, phi),signal) 
    signal[0] = signal[0]*np.exp(-TR/T2)
    signal[1] = signal[1]*np.exp(-TR/T2)
    signal[2] = M0+(signal[2]-M0)*np.exp(-TR/T1) 
    P = np.array([
            [ np.cos(beta*TE/TR),  np.sin(beta*TE/TR),   0],
            [-np.sin(beta*TE/TR),  np.cos(beta*TE/TR),   0],
            [            0,             0,   1]
            ])
    signal = np.matmul(P,signal)
    return signal

def vectorform_SSFP(M0, alpha, phi, dphi, beta, TR, TE, T1, T2, Nr):
    #M = np.asarray([0, 0, M0])
    
    '''
    -------------------------------------------------------------------------
    Parameters
    
    M0: array_like  
    Initial magnetization in the B0 field, function of the proton density rho represented by [x, y, z]  
    
    Alpha: radian 
    Flip or tip angle of the magnetization vector 
    
    phi: radian 
    Angle between the vector and x axis/ phase
    
    dphi: radian 
    Increment of phi
    
    TR: scalar in msec  
    Repition time of the pulse sequence 
    
    TE: scalar in msec  
    Echo time 
    
    T1: scalar in msec
    T1 value of the tissue
    
    T2: scalar in msec
    T2 value of the tissue
    
    Nr: scalar 
    Number of simulation   
    
    -------------------------------------------------------------------------
    Returns
    Signal : single complex value
    x component as real and y component as imag
    
    -------------------------------------------------------------------------
    Notes
    Neal's thesis chapter 3.2 equation 3.9
    The key is that in steady state, the M+ = M-, which allows to solve the equation
    
    -------------------------------------------------------------------------
    References
    
    [1] 
    Author: Dr Neal K Bangerter
    Title: Contrast enhancement and artifact reduction in steady state magnetic resonance imaging
    Link: https://www.proquest.com/openview/41a8dcfb0f16a1289210b3bd4f9ea82b/1.pdf?cbl=18750&diss=y&pq-origsite=gscholar
    '''

    assert alpha != 0, 'Only non zero numbers are allowed'
    assert T1 != 0, 'Only non zero numbers are allowed'
    assert T2 != 0, 'Only non zero numbers are allowed'
    assert TE != 0, 'Only non zero numbers are allowed'
    assert TR != 0, 'Only non zero numbers are allowed'
    assert TE <= TR, 'TE must be shorter than or equal to TR'
    
    M = M0
    I = np.identity(3)
    E = np.diag([np.exp(-TR/T2), np.exp(-TR/T2), np.exp(-TR/T1)])
    ETE = np.diag([np.exp(-TE/T2), np.exp(-TE/T2), np.exp(-TE/T1)])
    P = np.array([[np.cos(beta),np.sin(beta),0], [-np.sin(beta),np.cos(beta),0], [0,0,1]])
    PTE = np.array([[np.cos(beta*TE/TR), np.sin(beta*TE/TR), 0],
                   [-np.sin(beta*TE/TR), np.cos(beta*TE/TR), 0],
                   [                  0,                  0, 1]])
    R = rotMat(alpha, phi)
    #(I-P*E*Rx)^-1*(I-E)*M0     
    #Mneg = np.matmul(np.linalg.inv(I - np.matmul(P,np.matmul(E, Rx))), np.matmul(I-E, M0))
    
    #equation 3.7
    Mneg = np.linalg.inv((I-P@E@R))@(I-E)@M            
    
    #equation 3.8
    Mpos = R@Mneg                                      
    
    #equation 3.9
    MTE = PTE@ETE@Mpos + (I-ETE)@M 
    data = np.zeros([1], dtype = complex)
    data.real = MTE[0]
    data.imag = MTE[1]
    return data

def SSFP_trace(M0, alpha, phi, dphi, beta, TR, TE, T1, T2, Nr):
    signal = np.zeros([Nr,3])
    signal[0,:] = [0, 0, M0]
    signal[0,:] = np.matmul(signal[0,:],rotMat(alpha, phi)) 
    for i in range(1, Nr):
        signal[i,0] = signal[i-1,0]*np.exp(-TR/T2)
        signal[i,1] = signal[i-1,1]*np.exp(-TR/T2)
        signal[i,2] = M0+(signal[i-1,2]-M0)*np.exp(-TR/T1)
        P = np.array([
            [ np.cos(beta),  np.sin(beta),   0],
            [-np.sin(beta),  np.cos(beta),   0],
            [            0,             0,   1]
            ])
        signal[i,:] = np.matmul(signal[i,:],rotMat(alpha, phi))
        signal[i,:] = np.matmul(signal[i,:],P)
        phi = phi + dphi
        
    return signal

def rotMat(alpha, phi):
    rotation = np.array([
    [np.cos(alpha)*np.sin(phi)**2 + np.cos(phi)**2,          
         (1-np.cos(alpha))*np.cos(phi)*np.sin(phi),         
     -np.sin(alpha)*np.sin(phi)],
        
    [    (1-np.cos(alpha))*np.cos(phi)*np.sin(phi),        
       np.cos(alpha)*np.cos(phi)**2+np.sin(phi)**2,         
                        np.sin(alpha)*np.cos(phi)],
        
    [                    np.sin(alpha)*np.sin(phi),                      
                        -np.sin(alpha)*np.cos(phi),                    
                                     np.cos(alpha)]
    ])
    return rotation
