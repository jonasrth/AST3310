import numpy as np
import matplotlib.pyplot as plt

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
        F = r["pp"]/(2*r["33"] + r["34"])
        if 2*r["33"] + r["34"] > r["pp"]:
            r["33"] = F*r["33"]
            r["34"] = F*r["34"]

        # For the reactions following the 3He+4He reaction:
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



### CONSTANTS

rho_c = 1.62e5   # [kg/m^3] Sun core mass density
T_c = 1.57e7     # [K] Sun core temperature


### BULLET POINT 2:

def BulletPoint2():
    BP2 = EnergyProduction(rho_c,T_c)

    print("Reaction energy outputs")
    for i in BP2.Q:
        print("{}: {:.3f} MeV".format(i,BP2.Q[i]/BP2.MeV_J))

    Q_nu = BP2.Q_nu
    m = BP2.m
    u_conv = BP2.u_conv

    Q_tot = (4*m["1H"]-m["4He"])*u_conv

    # Percentage of energy lost to neutrinos:
    Pnu_PPI = 2*Q_nu["PP0"]/Q_tot
    Pnu_PPII = (Q_nu["PP0"]+Q_nu["PPII"])/Q_tot
    Pnu_PPIII = (Q_nu["PP0"]+Q_nu["PPIII"])/Q_tot
    Pnu_CNO = (Q_nu["CNO1"]+Q_nu["CNO2"])/Q_tot

    print("Percentage of energy lost to neutrinos for branches:")
    print("PPI: {:.2%}".format(Pnu_PPI))
    print("PPII: {:.2%}".format(Pnu_PPII))
    print("PPIII: {:.2%}".format(Pnu_PPIII))
    print("CNO: {:.2%}".format(Pnu_CNO))

### BULLET POINT 6

def BulletPoint6():

    # Plotting relative energy production rates for the PP branches
    # and CNO cycle as a function of temperature

    N = int(1e4)
    T = np.logspace(4,9,base=10,num=N)

    eps_PPI = np.zeros_like(T)
    eps_PPII = np.zeros_like(T)
    eps_PPIII = np.zeros_like(T)
    eps_CNO = np.zeros_like(T)
    for i in range(len(T)):
        iter = EnergyProduction(rho_c,T[i])
        tot = iter.eps["PPI"] + iter.eps["PPII"] + iter.eps["PPIII"] + iter.eps["CNO"]
        eps_PPI[i] = iter.eps["PPI"]/tot
        eps_PPII[i] = iter.eps["PPII"]/tot
        eps_PPIII[i] = iter.eps["PPIII"]/tot
        eps_CNO[i] = iter.eps["CNO"]/tot

    plt.plot(T,eps_PPI,label="PPI")
    plt.plot(T,eps_PPII,label="PPII")
    plt.plot(T,eps_PPIII,label="PPIII")
    plt.plot(T,eps_CNO,label="CNO")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Relative Energy Production")
    plt.show()


### BULLET POINT 7:

def BulletPoint7():

    N = int(5e4)
    E = np.linspace(1e-17,1e-13,N)

    BP7 = EnergyProduction(rho_c,T_c)

    k_B = 1.3806e-23
    e = 1.6022e-19
    eps0 = 8.8542e-12
    h = 6.6261e-34

    m = BP7.m

    # converting to kg
    for i in m:
        m[i] = m[i]*BP7.m_u

    # Initializing dictionary for atomic number
    Z = {}

    Z["1H"],Z["2H"],Z["3He"],Z["4He"] = 1,1,2,2
    Z["7Li"],Z["7Be"],Z["8Be"],Z["8B"] = 3,3,4,5
    Z["12C"],Z["13C"],Z["15O"] = 6,6,8
    Z["13N"],Z["14N"],Z["15N"] = 7,7,7


    def GamowPeak(E,T,p_i,p_k):

        """
        Returns the Gamow Peak at a given energy and temperature for
        particles p_a and p_b
        """

        # [kg] reduced mass:
        m_ = m[p_i]*m[p_k]/(m[p_i]+m[p_k])

        exp1 = np.exp(-E/(k_B*T))
        exp2 = np.exp(-np.sqrt(0.5*m_/E)*Z[p_i]*Z[p_k]*e**2*np.pi/(eps0*h))

        return exp1*exp2

    def PlotGP(E,T,p_i,p_k):
        """
        Returns normalized plot of the Gamow Peak of particles p_a and p_b,
        for a temperature T, over an energy interval E
        """
        GP = GamowPeak(E,T,p_i,p_k)
        dE = abs((E[-1]-E[0])/len(E))
        GP_int = np.sum(GP)*dE
        GP_norm = GP/GP_int
        plt.plot(E,GP_norm,label=p_i+"+"+p_k)

    PlotGP(E,T_c,"1H","1H")
    PlotGP(E,T_c,"2H","1H")
    PlotGP(E,T_c,"3He","3He")
    PlotGP(E,T_c,"4He","3He")
    PlotGP(E,T_c,"7Li","1H")
    PlotGP(E,T_c,"7Be","1H")

    PlotGP(E,T_c,"12C","1H")
    PlotGP(E,T_c,"13C","1H")
    PlotGP(E,T_c,"14N","1H")
    PlotGP(E,T_c,"15N","1H")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Energy [J]")
    plt.ylabel("P(Energy)")
    plt.show()

### TOTAL ENERGY OUTPUT:

def Etot():
    tot = EnergyProduction(rho_c,T_c)
    print("Total energy output")
    print("%.2e J/kg/s" % tot.eps_tot)


"""
COMMENT OUT IF YOU DON'T WANT EVERYTHING PRINTED AT ONCE
"""

# Returns energy output of every reaction as well as percentage
# of energy lost to neutrinos for PPI,PPII,PPIII and CNO
BulletPoint2()

# Returns plot of relative energy production for PPI,PPII,PPIII and CNO
BulletPoint6()

# Returns plot of Gamow Peaks for reactions in PP chain and CNO cycle
BulletPoint7()

# Returns total energy produced per second per kg
Etot()

# Change to "True" to print the sanity check values
SanityCheck = True

if SanityCheck == True:
    EnergyProduction(rho_c,T=1.57e7).SanityCheck()
    EnergyProduction(rho_c,T=1e8).SanityCheck()
