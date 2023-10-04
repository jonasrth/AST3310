import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as intp

def cross_section(R, L, F_C, show_every=20, sanity=False, savefig=False):
    """
    plot cross section of star
    :param R: radius, array
    :param L: luminosity, array
    :param F_C: convective flux, array
    :param show_every: plot every <show_every> steps
    :param sanity: boolean, True/False
    :param savefig: boolean, True/False
    """

    # R_sun = 6.96E8      # [m]
    R_sun = R[0]
    # L_sun = 3.846E26    # [W]
    L_sun = L[0]

    plt.figure(figsize=(800/100, 800/100))
    fig = plt.gcf()
    ax  = plt.gca()

    r_range = 1.2 * R[0] / R_sun
    rmax    = np.max(R)

    ax.set_xlim(-r_range, r_range)
    ax.set_ylim(-r_range, r_range)
    ax.set_aspect('equal')

    core_limit = 0.995 * L_sun

    j = 0
    for k in range(0,len(R)-1):
        j += 1
        # plot every <show_every> steps
        if j%show_every == 0:
            if L[k] >= core_limit:     # outside core
                if F_C[k] > 0.0:       # plot convection outside core
                    circle_red = plt.Circle((0, 0), R[k]/rmax, color='red', fill=False)
                    ax.add_artist(circle_red)
                else:                  # plot radiation outside core
                    circle_yellow = plt.Circle((0, 0), R[k]/rmax, color='yellow', fill=False)
                    ax.add_artist(circle_yellow)
            else:                      # inside core
                if F_C[k] > 0.0:       # plot convection inside core
                    circle_blue = plt.Circle((0, 0), R[k]/rmax, color='blue', fill=False)
                    ax.add_artist(circle_blue)
                else:                  # plot radiation inside core
                    circle_cyan = plt.Circle((0, 0), R[k]/rmax, color='cyan', fill=False)
                    ax.add_artist(circle_cyan)

    # create legends
    circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='red', fill=True)
    circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='yellow', fill=True)
    circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='blue', fill=True)
    circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color='cyan', fill=True)

    ax.legend([circle_red,circle_yellow,circle_cyan,circle_blue],\
              ['Convection outside core','Radiation outside core','Radiation inside core','Convection inside core'])
    plt.xlabel(r"$R/R^*$")
    plt.ylabel(r"$R/R^*$")
    plt.title('Cross section of star')
    plt.show()

    if savefig:
        if sanity:
            fig.savefig('Figures/sanity_cross_section.png', dpi=300)
        else:
            fig.savefig('Figures/final_cross_section.png', dpi=300)

class EnergyProduction:

    def __init__(self,rho,T,SanityCheck=False):

        ### Parameters:
        self.rho = rho          # [kg/m^3]
        self.T = T
        self.T9 = self.T*1e-9        # [1]

        ### [1] Mass fractions:
        self.X,self.Y_3He,self.Y = 0.7,1e-10,0.29
        self.Z_7Li,self.Z_7Be,self.Z_14N = 1e-7,1e-7,1e-11

        ### Constants:
        self.m_u = 1.660539e-27         # [kg] atomic mass unit
        N_A = 6.022141e23               # [1] Avagadros constant
        self.g_conv = 1/(1e6*N_A)       # [1] gamma conversion factor (to SI)
        self.MeV_J = 1e6*1.602176e-19   # [J/MeV] conversion factor from MeV to J
        self.u_conv = 931.4943          # [MeV/c^2/u] Conversion, u to MeV/c^2


        # [u] Atomic masses of isotopes:
        m = {}

        m["1H"],m["2H"],m["3He"],m["4He"] = 1.007825,2.014102,3.016029,4.002603
        m["7Li"],m["7Be"],m["8Be"],m["8B"] = 7.016003,7.016929,8.005305,8.024607
        m["12C"],m["13C"],m["15O"] = 12.000000,13.003355,15.003066
        m["13N"],m["14N"],m["15N"] = 13.005739,14.003074,15.000109

        self.m = m

        # [MeV] Energy of neutrinos in PP0,PPII,PPIII and CNO:
        Q_nu = {}

        Q_nu["PP0"],Q_nu["PPII"],Q_nu["PPIII"] = 0.265,0.815,6.711
        Q_nu["CNO1"],Q_nu["CNO2"] = 0.707,0.997

        self.Q_nu = Q_nu

        ### [m^-3] Number densities:
        self.n = self.NumberDensities()

        ### [m^3/s] Proportionality functions:
        self.lambda_ = self.ProportionalityFunctions()

        ### [1/kg/s] Reaction rates
        self.r = self.ReactionRates()

        ### [J] Reaction energy outputs:
        self.Q = self.ReactionOutputs()

        ### PP branch and CNO cycle energy production:

        Q = self.Q
        r = self.r

        Q_PP0 = Q["pp"] + Q["pd"]
        Q_CNO = Q["p12"] + Q["13"] + Q["p13"] + Q["p14"] + Q["15"] + Q["p15"]

        # [J/kg/s] Energy production rate
        self.eps = {}

        self.eps["PP0"] = r["pp"]*Q_PP0
        self.eps["PPI"] = r["33"]*(Q["33"] + 2*Q_PP0)
        self.eps["PPII"] = r["34"]*(Q["34"] + Q_PP0) \
        + r["e7"]*Q["e7"] + r["17_"]*Q["17_"]
        self.eps["PPIII"] = r["17"]*(Q["17"] + Q["decay"]) \
        + r["34"]*(Q["34"] + Q_PP0)
        self.eps["CNO"] = r["p14"]*Q_CNO

        ### [J/kg/s] Total energy production rate:

        self.eps_tot = r["pp"]*Q_PP0 + r["33"]*Q["33"] \
        + r["34"]*Q["34"] + r["e7"]*Q["e7"] + r["17_"]*Q["17_"] \
        + r["17"]*(Q["17"]+Q["decay"]) + r["p14"]*Q_CNO


    def SanityCheck(self):

        SC = np.zeros(7)

        r = self.r
        Q = self.Q
        Q_CNO = Q["p12"]+Q["13"]+Q["p13"]+Q["p14"]+Q["15"]+Q["p15"]

        SC[0] = r["pp"]*(Q["pp"] + Q["pd"])
        SC[1] = r["33"]*Q["33"]
        SC[2] = r["34"]*Q["34"]
        SC[3] = r["e7"]*Q["e7"]
        SC[4] = r["17_"]*Q["17_"]
        SC[5] = r["17"]*(Q["17"] + Q["decay"])
        SC[6] = r["p14"]*Q_CNO

        print(r"Sanity check for T = %.2e and rho = %.2e" % (self.T,self.rho))
        for i in range(len(SC)):
            print("%.4e Jm^-3s^-1" % (SC[i]*self.rho))

    def NumberDensities(self):

        """
        Returns number densities of elements
        """

        def NumDen(mass_frac,Z):
            n = mass_frac*self.rho/(Z*self.m_u)
            return n

        n = {}

        n["1H"] = NumDen(self.X,1)
        n["3He"] = NumDen(self.Y_3He,3)
        n["4He"] = NumDen(self.Y,4)
        n["7Li"] = NumDen(self.Z_7Li,7)
        n["7Be"] = NumDen(self.Z_7Be,7)
        n["14N"] = NumDen(self.Z_14N,14)

        n["e"] = n["1H"] + 2*n["3He"] + 2*n["4He"] \
        + 3*n["7Li"] + 3*n["7Be"] + 7*n["14N"]

        return n

    def ReactionRates(self):

        """
        Returns rates of reactions
        """

        lambda_ = self.lambda_

        def r_rate(p_i,p_k,lambda_ik):

            kronDel = 1 if p_i==p_k else 0
            r = self.n[p_i]*self.n[p_k]/(self.rho*(1 + kronDel))*lambda_ik
            return r

        r = {}

        r["pp"] = r_rate("1H","1H",lambda_["pp"])
        r["33"] = r_rate("3He","3He",lambda_["33"])
        r["34"] = r_rate("3He","4He",lambda_["34"])
        r["e7"] = r_rate("e","7Be",lambda_["e7"])
        r["17_"] = r_rate("7Li","1H",lambda_["17_"])
        r["17"] = r_rate("7Be","1H",lambda_["17"])
        r["p14"] = r_rate("14N","1H",lambda_["p14"])

        # Adjusting rates so no step consumes more elements than the previous
        # is able to produce:

        # For the reactions following the PP0 chain
        if r["33"] == 0 and r["34"] == 0:
            F = 0
        else:
            F = r["pp"]/(2*r["33"] + r["34"])
        if 2*r["33"] + r["34"] > r["pp"]:
            r["33"] = F*r["33"]
            r["34"] = F*r["34"]

        # For the reactions following the 3He+4He reaction:
        if r["e7"] == 0 and r["17"] == 0:
            F = 0
        else:
            F = r["34"]/(r["e7"] + r["17"])
        if r["e7"] + r["17"] > r["34"]:
            r["e7"] = F*r["e7"]
            r["17"] = F*r["17"]

        # For the reaction following the Beryllium electron capture
        if r["17_"] > r["e7"]:
            r["17_"] = r["e7"]

        return r


    def ReactionOutputs(self):

        """
        This function calculates the energy outputs for all the reactions
        in the PP-chain the dominant CNO cycle, excluding the energy carried
        away by neutrinos. Useful energy outputs are returned in units of
        Joules.
        """

        m = self.m              # [u] Isotope masses
        Q_nu = self.Q_nu        # [MeV] Neutrino energies
        u_conv = self.u_conv    # [MeV/c^2/u] Conversion, u to MeV/c^2
        u_conv = 931.494

        # [MeV] Calculating output energies for each reaction in the PP chain
        # and the dominant CNO cycle:

        def Q_(m_in,m_out,Q_nu=0):
            """
            Takes reactant and product mass as well as energy of a
            released neutrino and returns the released energy that
            contributes to the star's energy.
            """
            return (m_in - m_out)*u_conv - Q_nu

        # [MeV] Initializing dicionary for output energy
        Q = {}

        # PP0:
        Q["pp"] = Q_(2*m["1H"],m["2H"],Q_nu["PP0"])
        Q["pd"] = Q_(m["2H"]+m["1H"],m["3He"])
        # PPI:
        Q["33"] = Q_(2*m["3He"],m["4He"]+2*m["1H"])
        # PPII & PPIII:
        Q["34"] = Q_(m["3He"]+m["4He"],m["7Be"])
        # PPII:
        Q["e7"] = Q_(m["7Be"],m["7Li"],Q_nu["PPII"])
        Q["17_"] = Q_(m["7Li"]+m["1H"],2*m["4He"])
        # PPIII:
        Q["17"] = Q_(m["7Be"]+m["1H"],m["8B"])
        Q["decay"] = Q_(m["8B"],2*m["4He"],Q_nu["PPIII"])

        # CNO:
        Q["p12"] = Q_(m["12C"]+m["1H"],m["13N"])
        Q["13"] = Q_(m["13N"],m["13C"],Q_nu["CNO1"])
        Q["p13"] = Q_(m["13C"]+m["1H"],m["14N"])
        Q["p14"] = Q_(m["14N"]+m["1H"],m["15O"])
        Q["15"] = Q_(m["15O"],m["15N"],Q_nu["CNO2"])
        Q["p15"] = Q_(m["15N"]+m["1H"],m["12C"]+m["4He"])

        # Converting to Joules:
        for i in Q:
            Q[i] = self.MeV_J*Q[i]

        return Q

    def ProportionalityFunctions(self):

        """
        Returns dictionary containing the proportionality function (PF) for
        important reactions in the PP chain and CNO cycle
        """

        # Scaled temperature
        T9 = self.T9

        # Initializing dictionary for PFs
        lambda_ = {}

        # PF for 1H + 1H reaction:
        lambda_pp_1 = 4.01e-15*T9**(-2/3)*np.exp(-3.380*T9**(-1/3))
        lambda_pp_2 = 1 + 0.123*T9**(1/3) \
        + 1.09*T9**(2/3) + 0.938*T9

        lambda_["pp"] = self.g_conv*lambda_pp_1*lambda_pp_2

        # PF for 3He + 3He reaction:
        lambda_33_1 = 6.04e10*T9**(-2/3)*np.exp(-12.276*T9**(-1/3))
        lambda_33_2 = 1 + 0.034*T9**(1/3) - 0.522*T9**(2/3) \
        - 0.124*T9 + 0.353*T9**(4/3) + 0.213*T9**(5/3)

        lambda_["33"] = self.g_conv*lambda_33_1*lambda_33_2

        # PF for 3He + 4He reaction:
        T9_ = T9/(1 + 4.95e-2*T9)
        lambda_34 = 5.61e6*T9_**(5/6)*T9**(-3/2)*np.exp(-12.826*T9_**(-1/3))

        lambda_["34"] =  self.g_conv*lambda_34

        # PF for 7Be + e^- reaction:
        lambda_e7_1 = 1.34e-10*T9**(-1/2)
        lambda_e7_2 = 1 - 0.537*T9**(1/3) + 3.86*T9**(2/3) \
        + 0.0027*T9**(-1)*np.exp(2.515e-3*T9**(-1))

        lambda_e7 = lambda_e7_1*lambda_e7_2

        # Including upper limit for T < 10^6 K
        upper_limit = 1.57e-7/self.n["e"]
        if self.T < 1e6:
            if lambda_e7 > upper_limit:
                lambda_e7 = upper_limit

        lambda_["e7"] = self.g_conv*lambda_e7

        # PF for 7Li + 1H reaction:
        T9_ = T9/(1 + 0.759*T9)
        lambda_17 = 1.096e9*T9**(-2/3)*np.exp(-8.472*T9**(-1/3)) \
        - 4.830e8*T9_**(5/6)*T9**(-3/2)*np.exp(-8.472*T9_**(-1/3)) \
        + 1.06e10*T9**(-3/2)*np.exp(-30.442*T9**(-1))

        lambda_["17_"] = self.g_conv*lambda_17

        # PF for 7Be + 1H reaction:
        lambda_17 = 3.11e5*T9**(-2/3)*np.exp(-10.262*T9**(-1/3)) \
        + 2.53e3*T9**(-3/2)*np.exp(-7.306*T9**(-1))

        lambda_["17"] = self.g_conv*lambda_17

        # PF for 14N + 1H reaction:
        lambda_p14_1 = 4.90e7*T9**(-2/3)*np.exp(-15.228*T9**(-1/3) - 0.092*T9**2)
        lambda_p14_2 = 1 + 0.027*T9**(1/3) - 0.778*T9**(2/3) \
        - 0.149*T9 + 0.261*T9**(4/3) + 0.127*T9**(5/3)
        lambda_p14_3 = 2.37e3*T9**(-3/2)*np.exp(-3.011*T9**(-1)) \
        + 2.19e4*np.exp(-12.53*T9**(-1))

        lambda_["p14"] = self.g_conv*(lambda_p14_1*lambda_p14_2 + lambda_p14_3)


        return lambda_

class StarModel:

    def __init__(self,
        L0 = 1,         # [W] Star luminosity in solar luminosities
        r0 = 1,         # [m] Star radius in solar radii
        m0 = 1,         # [kg] Star mass in solar masses
        rho0 = 1.42e-7, # [kg/m^3] Star surface density in av. solar density
        T0 = 5770       # [K] Star surface temperature
        ):

        ### CONSTANTS

        # Solar parameters:
        self.L_sun = 3.846e26    # [W] Solar luminosity
        self.R_sun = 6.96e8      # [m] Solar radius
        self.M_sun = 1.989e30    # [kg] Solar mass
        self.Rho_sun = 1.408e3   # [kg/m^3] Average density of the sun

        # [1] Mass fractions
        X = 0.7             # Hydrogen
        Y_3He = 1e-10       # Helium-3
        Y = 0.29            # Helium
        Z_7Li = 1e-7        # Lithium
        Z_7Be = 1e-7        # Beryllium
        Z_14N = 1e-11       # Nitrogen

        # Mean molecular weight
        self.mu = 1/(2*X + 3*Y_3He/3 + 3*Y/4
        + 4*Z_7Li/7 + 5*Z_7Be/7 + 8*Z_14N/14)

        # General constants:
        self.G = 6.6742e-11         # [Nm^2kg^(-2)] Gravitational constant
        self.kB = 1.3806e-23        # [J/K] Boltzmann's constant
        self.sigma = 5.6704e-8      # [Wm^(-2)K^(-4)] Stefan-Boltzmann constant
        self.c = 2.9979e8           # [m/s] Speed of light
        self.m_u = 1.6605e-27       # [kg] Atomic mass unit

        # [J/K/kg] Heat capacity
        self.cp = (5/2)*self.kB/(self.mu*self.m_u)


        # Interpolating data
        self.logR,self.logT,self.logK = self.readfile("opacity.txt")
        self.IP = intp.interp2d(self.logR,self.logT,self.logK)
        self.warning = False

        ### Initializing

        self.L = L0*self.L_sun
        self.r = r0*self.R_sun
        self.m = m0*self.M_sun
        self.rho = rho0*self.Rho_sun
        self.T = T0

        # [N/m^2] Pressure:
        self.P_gas = self.kB*self.T/(self.mu*self.m_u)*self.rho
        self.P_rad = 4*self.sigma/(3*self.c)*self.T**4
        self.P = self.P_gas + self.P_rad

        # Free parameter, can take values between 1/2 and 2
        self.alpha = 1

        # Initializing useful state-dependent quantities
        self.Quantities()

    def Quantities(self):

        """
        Useful state-dependent quantities
        """

        # [N/m^2] Pressure:
        self.P_rad = 4*self.sigma/(3*self.c)*self.T**4
        self.P_gas = self.P - self.P_rad

        # [kg/m^3] Mass density
        self.rho = self.mu*self.m_u/(self.kB*self.T)*self.P_gas

        # Class from project 1, contains info about energy production
        self.EP = EnergyProduction(self.rho,self.T)

        # [m^2/kg] Opacity
        self.K = self.findK(self.rho,self.T)

        # [m] Scale height:
        #self.Hp = self.kB*self.T/(self.mu*self.m_u)*self.r**2/(self.G*self.m)
        self.Hp = -self.P*(self.drdm(self.m,self.r)/self.dPdm(self.m,self.P))

        # [m] Mixing length:
        self.lm = self.alpha*self.Hp

        # [m/s^2] Gravitational acceleration:
        self.g = self.G*self.m/self.r**2

        # [m^2] To simplify later expressions:
        self.U = 64*self.sigma*self.T**3/(3*self.K*self.rho**2*self.cp)\
        *np.sqrt(self.Hp/self.g)

        ### [1] Temperature gradients:

        self.nabla_ad = 2/5
        self.nabla_stable = 3*self.L*self.K*self.rho\
        *self.Hp/(64*np.pi*self.r**2*self.sigma*self.T**4)

        self.eta = self.find_eta()

        self.nabla_star = -self.lm**2/self.U*self.eta**3 + self.nabla_stable
        self.nabla_p = self.nabla_star - self.eta**2

        if self.nabla_stable > self.nabla_ad:
            self.nabla_star += 0
            self.F_con = self.rho*self.cp*self.T\
            *np.sqrt(self.g/self.Hp**3)*(self.lm/2)**2*self.eta**3
        else:
            self.nabla_star = self.nabla_stable
            self.F_con = 0

        ### [W] Energy flux:

        # Radiative flux:
        self.F_rad = 16*self.sigma*self.T**4/(3*self.K*self.rho*self.Hp)\
        *self.nabla_star

        """
        # Convective flux:
        self.F_con = self.rho*self.cp*self.T\
        *np.sqrt(self.g/self.Hp**3)*(self.lm/2)**2*self.eta**3
        """

    def readfile(self,filename):

        """
        Extracting logT,logR and correspodning logK data from opacity.txt file
        """

        file = open(filename,"r")
        line = file.readline()

        logR = np.asarray(line.split()[1:],dtype=float)

        file.readline()

        lines = file.readlines()

        logT = np.zeros(len(lines))
        logK = np.zeros((len(lines),len(logR)))

        for i in range(len(lines)):
            line = np.asarray(lines[i].split(),dtype=float)
            logT[i] = line[0]
            logK[i,:] = line[1:]

        return logR,logT,logK

    def findK(self,rho,T):

        """
        Takes density rho and temperature T in SI units and returns
        opacity K in SI units by interpolating data from opacity.txt file
        """

        logT = np.log10(T)
        rho_cgs = rho*1e-3
        R = rho_cgs/(T*1e-6)**3
        logR = np.log10(R)

        if logT < np.min(self.logT) or logT > np.max(self.logT)\
        or logR < np.min(self.logR) or logR > np.max(self.logR):
            if self.warning == False:
                print("WARNING: T and/or rho outside of bounds of opacity table")
                self.warning = True


        # Finding the log of the opacity
        logK = self.IP(logR,logT)[0]

        # Removing the logarithm:
        K_cgs = 10**(logK)
        # Converting from cgs to SI:
        K = K_cgs*1e-1

        return K

    def find_eta(self):

        """
        Finding eta = sqrt(nabla_star-nabla_packet) by solving
        third degree polymonial and returning the real root.
        """

        a = self.lm**2/self.U
        b = 1
        c = self.U*(4/self.lm**2)
        d = - (self.nabla_stable - self.nabla_ad)

        eta = np.roots([a,b,c,d])

        return eta[np.argwhere(eta.imag==0)].real[0][0]


    def RungeKutta4(self,dydm,y):

        h = self.dm

        k1 = dydm(self.m, y)
        k2 = dydm(self.m + h/2, y + h*k1/2)
        k3 = dydm(self.m + h/2, y + h*k2/2)
        k4 = dydm(self.m + h, y + h*k3)

        y_new = y + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)

        return y_new

    def drdm(self,m,r):
        return 1/(4*np.pi*r**2*self.rho)

    def dPdm(self,m,P):
        return -self.G*m/(4*np.pi*self.r**4)

    def dLdm(self,m,L):
        return self.EP.eps_tot

    def dTdm(self,m,T):

        if self.nabla_stable > self.nabla_ad:
            dTdm = self.nabla_star*(T/self.P)*self.dPdm(m,self.P)
        else:
            #dTdm = self.nabla_stable*(T/self.P)*self.dPdm(m,self.P)
            dTdm = -3*self.K*self.L/(256*np.pi**2*self.sigma*self.r**4*T**3)

        return dTdm

    def step_size(self,p):

        dm_r = p*self.r/self.drdm(self.m,self.r)
        dm_P = p*self.P/self.dPdm(self.m,self.P)
        dm_L = p*self.L/self.dLdm(self.m,self.L)
        dm_T = p*self.T/self.dTdm(self.m,self.T)

        dm = np.min(abs(np.array([dm_r,dm_P,dm_L,dm_T])))

        return dm


    def step(self,dm):

        """
        Updates class for one step dm towards star center
        """

        self.dm = -dm


        Solver = self.RungeKutta4
        #Solver = self.ForwardEuler

        # New variables after state change:

        self.r = Solver(self.drdm,self.r)
        self.P = Solver(self.dPdm,self.P)
        self.L = Solver(self.dLdm,self.L)
        self.T = Solver(self.dTdm,self.T)

        self.m = self.m + self.dm

        self.Quantities()

    def SanityCheck(self):

        """
        Sanity check 1: Opacity table
        """

        # arrays of values to check
        logT = np.array([3.750,3.755,3.755,3.755,3.755,3.770,
        3.780,3.795,3.770,3.775,3.780,3.795,3.800])
        logR = -np.array([6.00,5.95,5.80,5.70,5.55,5.95,5.95,
        5.95,5.80,5.75,5.70,5.55,5.50])

        T = 10**logT
        rho_cgs = (T*1e-6)**3*10**logR
        rho = rho_cgs*1e3

        print("Opacity Table Sanity Check:\
        \nlogT\tlogR[cgs]   logK[cgs]\tK[SI]")
        for i in range(len(logT)):
            K = self.findK(rho[i],T[i])
            K_cgs  = 10*K
            logK = np.log10(K_cgs)
            print("{:.3f}\t{:.2f}".format(logT[i],logR[i]) +
            "\t    {:.2f}\t{:.2e}".format(logK,K))

        print("\n")

        """
        Sanity check 2: Example 5.1
        """

        rho0 = 55.9
        Rho_sun = self.Rho_sun
        K = 3.98

        from types import MethodType

        SC2 = StarModel(r0 = 0.84,m0 = 0.99,rho0=rho0/Rho_sun,T0 = 0.9e6)

        new_findK = lambda self,rho,T: K
        SC2.findK = MethodType(new_findK,SC2)

        SC2.Quantities()

        print("Example 5.1 Sanity Check:")
        print("nabla_stable        = {:.2f}".format(SC2.nabla_stable))
        print("nabla_ad            = {:}".format(SC.nabla_ad))
        print("Hp                  = {:.2e} m".format(SC2.Hp))
        print("U                   = {:.2e} m^2".format(SC2.U))
        print("eta                 = {:.3e}".format(SC2.eta))
        print("nabla_star          = {:.3f}".format(SC2.nabla_star))

        F_con = SC2.F_con
        F_rad = SC2.F_rad

        print("F_con/(F_con+F_rad) = {:.2f}".format(F_con/(F_rad+F_con)))
        print("F_rad/(F_con+F_rad) = {:.2f}".format(F_rad/(F_rad+F_con)))

        print("nabla_ad < nabla_p   < nabla_star   < nabla_stable:")
        print("{:}      < {:.6f}  < {:.6f}     < {:.2f}".format(SC2.nabla_ad,\
        SC2.nabla_p,SC2.nabla_star,SC2.nabla_stable))

        """
        Sanity check 3: Temperature gradient plot
        """

        print("\nSaniy check 3 and 4 (figures):")

        SC3 = StarModel()

        param,F,eps,nabla = self.Simulate(SC3,nabla=True)

        m,r,rho,P,L,T = param
        F_con,F_rad = F
        nabla_ad,nabla_star,nabla_stable = nabla

        plt.plot(r/r[0],nabla_stable)
        plt.plot(r/r[0],nabla_star)
        plt.plot(r/r[0],nabla_ad)
        plt.legend([r"$\nabla_{stable}$",r"$\nabla^*$",\
        r"$\nabla_{ad}$"])
        plt.yscale("symlog")
        plt.xlabel(r"$R/R_\odot$")
        plt.ylabel(r"$\nabla$")
        plt.show()

        """
        Sanity check 4: Cross section of star
        """

        cross_section(r,L,F_con,show_every=5,sanity=True,savefig=True)



    def Simulate(self,init,p=0.01,eps=False,nabla=False):

        """
        Moves inward into star and logs the values of different parameters so
        we can plot them
        """

        m=r=rho=P=L=T=F_con=F_rad= np.array([])
        eps_tot=eps_PPI=eps_PPII=eps_PPIII=eps_CNO = np.array([])
        nabla_ad=nabla_star=nabla_stable = np.array([])


        found = False

        while init.m > 0 and init.r > 0:

            m = np.append(m,init.m)
            r = np.append(r,init.r)
            rho = np.append(rho,init.rho)
            P = np.append(P,init.P)
            L = np.append(L,init.L)
            T = np.append(T,init.T)

            F_con = np.append(F_con,init.F_con)
            F_rad = np.append(F_rad,init.F_rad)

            if eps:
                eps_tot = np.append(eps_tot,init.EP.eps_tot)
                eps_PPI = np.append(eps_PPI,init.EP.eps["PPI"])
                eps_PPII = np.append(eps_PPII,init.EP.eps["PPII"])
                eps_PPIII = np.append(eps_PPIII,init.EP.eps["PPIII"])
                eps_CNO = np.append(eps_CNO,init.EP.eps["CNO"])

            if nabla:
                nabla_ad = np.append(nabla_ad,init.nabla_ad)
                nabla_star = np.append(nabla_star,init.nabla_star)
                nabla_stable = np.append(nabla_stable,init.nabla_stable)


            # Calculating next step

            dm = init.step_size(p)
            init.step(dm)


            # finding where core starts
            if L[-1]>0.995*L[0] and init.L<0.995*L[0]:
                print("Core starts at {:.2f}R\n".format(init.r/r[0]))
            # finding width of upper convection layer
            if F_con[-1]!=0 and init.F_con==0 and found==False:
                print("Upper convection zone of width {:.2f}R"\
                .format((r[0]-init.r)/r[0]))
                found = True

            # prevents loop from going on forever
            if init.r==r[-1]:
                break

        param = (m,r,rho,P,L,T)
        F = (F_con,F_rad)
        nabla = (nabla_ad,nabla_star,nabla_stable)
        eps = (eps_tot,eps_PPI,eps_PPII,eps_PPIII,eps_CNO)

        return param,F,eps,nabla



"""
To view the sanity checks set the "SanityCheck" variable under to True
"""

SanityCheck = True

if SanityCheck:
    SC = StarModel()
    SC.SanityCheck()

"""
Set statements below to true if you want to run functions
"""

param_test = True


def parameter_test():

    """
    Testing different values of the parameters r0,rho0,T0 and P0:
    """

    print("With parameters given in project text:")

    Test = StarModel()
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)


    ### Testing r0

    print("Decreasing radius:")
    Test = StarModel(r0 = 1/2)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)

    print("Increasing radius:")
    Test = StarModel(r0 = 2)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)


    ### Testing T0:

    print("Decreasing surface temperature:")
    Test = StarModel(T0 = 4000)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)

    print("Increasing surface temperature:")
    Test = StarModel(T0 = 7000)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)


    ### Testing rho0

    rho0 = 1.42e-7

    print("Decreasing surface density:")
    Test = StarModel(rho0 = 0.1*rho0)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)

    print("Increasing surface density:")
    Test = StarModel(rho0 = 10*rho0)
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)


    ### Testing P0:

    print("Decreasing surface pressure:")
    Test = StarModel()
    Test.P *= 0.1
    Test.Quantities()
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)

    print("Increasing surface pressure:")
    Test = StarModel()
    Test.P *= 10
    Test.Quantities()
    param,F,eps,nabla = Test.Simulate(Test)
    m,r,rho,P,L,T = param
    F_con,F_rad = F
    cross_section(r,L,F_con,show_every=10)

if param_test:
    parameter_test()


"""
Final model of our star:
"""

plot_main = True   # Set to True to plot main parameters
plot_flux = True    # Set to True to plot relative flux
plot_eps = True    # Set to True to plot relative energy production
plot_nabla = True   # Set to True to plot the temperature gradients
plot_cross = True   # Set to True to plot the star's cross section


Model = StarModel(r0 = 2,rho0 = 10*1.42e-7)

param,F,eps,nabla = Model.Simulate(Model,eps=True,nabla=True)

m,r,rho,P,L,T = param
F_con,F_rad = F
eps_tot,eps_PPI,eps_PPII,eps_PPIII,eps_CNO = eps
nabla_ad,nabla_star,nabla_stable = nabla

def plot_main_params():

    """
    ### Plotting main parameters as function of radius:
    """

    fig, axs = plt.subplots(5)
    #fig.suptitle('Main parameters as function of radius')
    axs[0].plot(r/r[0], m/m[0])
    axs[1].plot(r/r[0], T)
    axs[2].plot(r/r[0], L/L[0])
    axs[3].plot(r/r[0], rho)
    axs[4].plot(r/r[0], P)

    ylabel = [r"m/$M_\odot$","T",r"$L/L_\odot$",r"$\rho$","P"]

    i = 0
    for ax in axs:
        ax.set(xlabel=r'$R/R^*$', ylabel=ylabel[i])
        if i > 2:
            ax.set(yscale="log")
        i += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs:
        ax.label_outer()

    plt.show()

def plot_relative_flux():

    """
    ### Plotting relative flux:
    """

    F_tot = F_con + F_rad
    plt.plot(r/r[0],F_con/F_tot,label=r"$F_{con}$")
    plt.plot(r/r[0],F_rad/F_tot,label=r"$F_{rad}$")
    plt.legend()
    plt.xlabel(r"$R/R^*$")
    plt.ylabel("Relative flux")
    plt.show()

def plot_relative_energy_production():

    """
    ### Plotting relative energy production:
    """

    eps = eps_PPI + eps_PPII + eps_PPIII + eps_CNO

    plt.plot(r/r[0],eps_PPI/eps,label=r"$\epsilon_{PPI}$")
    plt.plot(r/r[0],eps_PPII/eps,label=r"$\epsilon_{PPII}$")
    plt.plot(r/r[0],eps_PPIII/eps,label=r"$\epsilon_{PPIII}$")
    plt.plot(r/r[0],eps_CNO/eps,label=r"$\epsilon_{CNO}$")
    plt.plot(r/r[0],eps_tot/np.max(eps_tot),label=r"$\epsilon(r)/\epsilon_{max}$")
    plt.legend()
    plt.xlabel(r"$R/R^*$")
    plt.ylabel("Relative energy production")
    plt.show()

def plot_temp_gradients():

    """
    ### Plotting temperature gradients:
    """

    plt.plot(r/r[0],nabla_stable)
    plt.plot(r/r[0],nabla_star)
    plt.plot(r/r[0],nabla_ad)
    plt.legend([r"$\nabla_{stable}$",r"$\nabla^*$",\
    r"$\nabla_{ad}$"])
    plt.yscale("symlog")
    plt.xlabel(r"$R/R^*$")
    plt.ylabel(r"$\nabla$")
    plt.show()


if plot_main:
    plot_main_params()

if plot_flux:
    plot_relative_flux()

if plot_eps:
    plot_relative_energy_production()

if plot_nabla:
    plot_temp_gradients()

if plot_cross:
    cross_section(r,L,F_con,show_every=5,savefig=True)
