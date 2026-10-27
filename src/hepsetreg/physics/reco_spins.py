import awkward as ak
import vector 
import numpy as np

def boost(top,anti_top,leptonP,leptonM):

    """
    Performs the relevant boosts and scalar products to return the cos-variables
    """

    # Build ttbar system
    ttbar = top + anti_top 

    # Boost tops lab --> ttbar_CoM
    top_in_CoM      = top.boostCM_of(ttbar)
    antitop_in_CoM  = anti_top.boostCM_of(ttbar) 

    # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
    leptonP_in_CoM     = leptonP.boostCM_of(ttbar)
    leptonM_in_CoM     = leptonM.boostCM_of(ttbar)
    leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
    leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

    # Compute the unit 3-vector directions
    Kdirection        = top_in_CoM.to_beta3().unit()
    leptonP_direction = leptonP_in_top.to_beta3().unit()
    leptonM_direction = leptonM_in_antitop.to_beta3().unit()

    return Kdirection, leptonP_direction, leptonM_direction


def helicity_basis_observables(Kdirection, leptonP_direction, leptonM_direction):

    """
    The helicity basis is a basis used in the measurement of spin observables at
    the LHC
    """

    # Kinematics
    z     = vector.obj(x=0,y=0,z=1)
    cos_T = z.dot(Kdirection)
    sin_T = (1 - cos_T**2)**0.5
    mask = 1*(cos_T > 0) -1*(cos_T < 0)

    # Helicity basis observables
    cos_K_plus  = Kdirection.dot(leptonP_direction)
    cos_K_minus = -Kdirection.dot(leptonM_direction) 

    Ndirection = (z.cross(Kdirection)/sin_T)
    cos_N_plus  = mask*Ndirection.dot(leptonP_direction)
    cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 

    Rdirection  =  1/sin_T * (z - cos_T*Kdirection)
    cos_R_plus  = mask*Rdirection.dot(leptonP_direction)
    cos_R_minus = -mask*Rdirection.dot(leptonM_direction) 

    cos_phi         = leptonP_direction.dot(leptonM_direction)
    cos_theta_star  =  cos_T

    # C_han: same construction as C_hel (cos_phi = leptonP.leptonM, the
    # opening-angle observable), but with leptonM_direction reflected
    # through the plane perpendicular to the helicity axis K first -- i.e.
    # its K-component sign is flipped, its transverse (R,N) part is kept.
    # This is the D_3 observable of Severi, Degli Esposti, Maltoni & Sioli,
    # arXiv:2205.00542 (eq. 30-33): cos_theta'_ab = p_hat_a . P3 p_hat_b
    # with P3 = diag(1,1,-1) acting in the (R,N,K) frame, measuring the
    # trace combination C_RR + C_NN - C_KK instead of C_hel's
    # C_KK + C_RR + C_NN.
    leptonM_reflected = leptonM_direction - 2*leptonM_direction.dot(Kdirection)*Kdirection
    cos_han         = leptonP_direction.dot(leptonM_reflected.unit())

    helicity_observables = ak.zip({
    "cos_K_plus"  :  cos_K_plus,
    "cos_K_minus" :  cos_K_minus,
    "cos_N_plus"  :  cos_N_plus,
    "cos_N_minus" :  cos_N_minus,
    "cos_R_plus"  :  cos_R_plus,
    "cos_R_minus" :  cos_R_minus,
    "cos_phi"     :  cos_phi,
    "cos_han"     :  cos_han,
    "cos_theta_star" : cos_theta_star
    }, depth_limit=1, with_name="Event")

    return helicity_observables