#ifndef INCLUDED_CONSTANTS_H
#define INCLUDED_CONSTANTS_H

////////////physical units//////////////
extern double c; //[cm/s]
extern double eV,GeV,pc; //[cm/pc]//
extern double G;//[cm^3 g^-1 s^-2]//
extern double M_sun;
extern double Mpc,kpc; //[cm]//
extern double Gyr;
extern double planck_const_erg_s;

extern double kB; //[eV/K]//
extern double mn,mp,me,millibarn_cm2,arcmin,arcsec,sigma_Thomson;
extern double Jy, mJy;

extern double pion0_mass_gev, pion_charged_mass_gev, muon_mass_gev;

/* Legacy compatibility declarations. Prefer canonical names above. */
extern double hPl;
extern double mb;
extern double mpi0,mpi_pm, mmu;

#endif
