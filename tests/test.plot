*******************************************************************************
* test.plot — WRPLOT reference spectra for the Python wrplot.py test suite
*
* Run from the tests/ directory:
*   wrplot test.plot +
* or:
*   ./run_wrplot_refs.sh
*
* Writes reference tables into ref/.
* Input continua live in data/.
*
* ISMLINE uses two Si III λ1206 components (GAL + SMC).
*******************************************************************************

 PAPERFORMAT A4Q BBROTATE
\FONT=HELVET
\SET_NDATMAX = 600000


*******************************************************************************
* PLOT 1 — ISMLINE only (GAL + SMC Si III λ1206)
*******************************************************************************
 PLOT   : ref ISMLINE SiIII 1206 GAL+SMC

\INSTRUCTION EXPORT
\VAR EBV = 0.14
\VAR EBV_GAL = 0.06
\CALC EBV_SMC = $EBV - $EBV_GAL
\CALC COLDENS_H_GAL = 1.25 * $EBV * 6.86e21
\CALC COLDENS_H_SMC = $EBV_SMC * 3.8e21 * 6
\CALC NEGCOLDENS_H_GAL = -1*$COLDENS_H_GAL
\CALC NEGCOLDENS_H_SMC = -1*$COLDENS_H_SMC
\CALC CD_S_GAL = $NEGCOLDENS_H_GAL * 3.55e-5/1000
\CALC CD_S_SMC = $NEGCOLDENS_H_SMC * 3.55e-5/1000
\VAR VRAD_SMC = 0
\VAR VRAD_GAL = -156.
\VAR VTURB_SMC = 30
\VAR VTURB_GAL = 10
\ECHO CD_S_GAL=$CD_S_GAL CD_S_SMC=$CD_S_SMC
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\ISMLINE reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1170.      1250.          50        100        0.
Y:   14CM          0        2         2        5        0.

N=? XYTABLE COLOR=2 SYMBOL 5 PEN=4
COMMAND ISMLINE L0=1206.51 COLDENS=$CD_S_GAL GAMM=6.1e9 F=1.67 A=28 VTURB=$VTURB_GAL VRAD=$VRAD_GAL
COMMAND ISMLINE L0=1206.51 COLDENS=$CD_S_SMC GAMM=6.1e9 F=1.67 A=28 VTURB=$VTURB_SMC VRAD=$VRAD_SMC
COMMAND WRITE FILE=ref/ref_ismline.txt
COMMAND INCLUDE data/continuum_iue.dat

END


*******************************************************************************
* PLOT 2 — ISMLINE (GAL+SMC) + CONVOL GAUSS=0.5
*******************************************************************************
 PLOT   : ref ISMLINE + CONVOL GAUSS=0.5

\INSTRUCTION EXPORT
\VAR EBV = 0.14
\VAR EBV_GAL = 0.06
\CALC EBV_SMC = $EBV - $EBV_GAL
\CALC COLDENS_H_GAL = 1.25 * $EBV * 6.86e21
\CALC COLDENS_H_SMC = $EBV_SMC * 3.8e21 * 6
\CALC NEGCOLDENS_H_GAL = -1*$COLDENS_H_GAL
\CALC NEGCOLDENS_H_SMC = -1*$COLDENS_H_SMC
\CALC CD_S_GAL = $NEGCOLDENS_H_GAL * 3.55e-5/1000
\CALC CD_S_SMC = $NEGCOLDENS_H_SMC * 3.55e-5/1000
\VAR VRAD_SMC = 0
\VAR VRAD_GAL = -156.
\VAR VTURB_SMC = 30
\VAR VTURB_GAL = 10
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\ISMLINE+CONVOL reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1170.      1250.          50        100        0.
Y:   14CM          0        2         2        5        0.

N=? XYTABLE COLOR=2 SYMBOL 5 PEN=4
COMMAND ISMLINE L0=1206.51 COLDENS=$CD_S_GAL GAMM=6.1e9 F=1.67 A=28 VTURB=$VTURB_GAL VRAD=$VRAD_GAL
COMMAND ISMLINE L0=1206.51 COLDENS=$CD_S_SMC GAMM=6.1e9 F=1.67 A=28 VTURB=$VTURB_SMC VRAD=$VRAD_SMC
COMMAND SORT JOIN
COMMAND CONVOL GAUSS=0.5
COMMAND WRITE FILE=ref/ref_ismline_convol.txt
COMMAND INCLUDE data/continuum_iue.dat

END


*******************************************************************************
* PLOT 3 — HLYMANA COLDENS=-1e20 on flat continuum
*******************************************************************************
 PLOT   : ref HLYMANA N=-1e20 flat

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\HLYMANA reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           900.      1300.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND HLYMANA COLDENS=-1.E20
COMMAND WRITE FILE=ref/ref_hlymana_1e20.txt
COMMAND INCLUDE data/continuum_lyman_flat.dat

END


*******************************************************************************
* PLOT 4 — HLYMANA COLDENS=-5e20 on flat continuum
*******************************************************************************
 PLOT   : ref HLYMANA N=-5e20 flat

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\HLYMANA reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           900.      1300.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND HLYMANA COLDENS=-5.E20
COMMAND WRITE FILE=ref/ref_hlymana_5e20.txt
COMMAND INCLUDE data/continuum_lyman_flat.dat

END


*******************************************************************************
* PLOT 5 — HLYMANA COLDENS=-1e21 on flat continuum
*******************************************************************************
 PLOT   : ref HLYMANA N=-1e21 flat

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\HLYMANA reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           900.      1300.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND HLYMANA COLDENS=-1.E21
COMMAND WRITE FILE=ref/ref_hlymana_1e21.txt
COMMAND INCLUDE data/continuum_lyman_flat.dat

END


*******************************************************************************
* PLOT 6 — HLYMANA on real UV continuum (windowed)
*******************************************************************************
 PLOT   : ref HLYMANA N=-1e20 on UV spectrum

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\HLYMANA on UV continuum
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           900.      1300.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND HLYMANA COLDENS=-1.E20
COMMAND WRITE FILE=ref/ref_hlymana_uv_1e20.txt
COMMAND INCLUDE data/uv_spectrum.dat

END


*******************************************************************************
* PLOT 7 — CONVOL GAUSS on synthetic absorption spectrum
*******************************************************************************
 PLOT   : ref CONVOL GAUSS=2.0 synthetic

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\CONVOL GAUSS reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           4800.     5200.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND CONVOL GAUSS=2.0
COMMAND WRITE FILE=ref/ref_convol_gauss.txt
COMMAND INCLUDE data/continuum_flat.dat

END


*******************************************************************************
* PLOT 8 — CONVOL BOX (KASTEN) on synthetic absorption spectrum
*******************************************************************************
 PLOT   : ref CONVOL KASTEN=2.0 synthetic

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\CONVOL BOX reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           4800.     5200.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND CONVOL KASTEN=2.0
COMMAND WRITE FILE=ref/ref_convol_box.txt
COMMAND INCLUDE data/continuum_flat.dat

END


*******************************************************************************
* PLOT 9 — CONVOL ROTATION on synthetic absorption spectrum
*******************************************************************************
 PLOT   : ref CONVOL ROTATION=2.5 BETA=0.6 synthetic

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\CONVOL ROT reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           4800.     5200.          50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND CONVOL ROTATION=2.5 BETA=0.6
COMMAND WRITE FILE=ref/ref_convol_rot.txt
COMMAND INCLUDE data/continuum_flat.dat

END


*******************************************************************************
* PLOT 10 — REDDENING SEATON EBV=1 on flat continuum (1000–1e5 Å)
* Input Y=0 means log10(flux)=0. After XLOG + REDDENING + XDEX + YDEX the
* WRITE table is linear λ vs reddened linear flux.
*******************************************************************************
 PLOT   : ref REDDENING SEATON EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING SEATON reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 SEATON
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_seaton.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 11 — REDDENING CARDELLI EBV=1 (default Rv=3.1)
*******************************************************************************
 PLOT   : ref REDDENING CARDELLI EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING CARDELLI reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 CARDELLI
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_cardelli.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 12 — REDDENING CARDELLI EBV=1 Rv=5.0
*******************************************************************************
 PLOT   : ref REDDENING CARDELLI EBV=1 Rv=5

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING CARDELLI Rv=5 reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 CARDELLI 5.0
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_cardelli_rv5.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 13 — REDDENING FITZPATRICK EBV=1 (default Rv=3.1)
*******************************************************************************
 PLOT   : ref REDDENING FITZPATRICK EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING FITZPATRICK reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 FITZPATRICK
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_fitzpatrick.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 14 — REDDENING FITZPATRICK EBV=1 Rv=5.0 (IR Moneti scales with Rv)
*******************************************************************************
 PLOT   : ref REDDENING FITZPATRICK EBV=1 Rv=5

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING FITZPATRICK Rv=5 reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 FITZPATRICK 5.0
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_fitzpatrick_rv5.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 15 — REDDENING LMC EBV=1
*******************************************************************************
 PLOT   : ref REDDENING LMC EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING LMC reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 LMC
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_lmc.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 16 — REDDENING SMC EBV=1 (default Rv=2.7)
*******************************************************************************
 PLOT   : ref REDDENING SMC EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING SMC reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 SMC
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_smc.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END


*******************************************************************************
* PLOT 17 — REDDENING SMC_BAR EBV=1
*******************************************************************************
 PLOT   : ref REDDENING SMC_BAR EBV=1

\INSTRUCTION EXPORT
\INSTRUCTION NO-EXPORT

HEADER :\CENTER\REDDENING SMC_BAR reference
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Flux
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:   23CM           1000.    100000.        50        100        0.
Y:   14CM          0        1.5       2        5        0.

N=? XYTABLE COLOR=2
COMMAND XLOG
COMMAND REDDENING 1 SMC_BAR
COMMAND XDEX
COMMAND YDEX
COMMAND WRITE FILE=ref/ref_reddening_smc_bar.txt
COMMAND INCLUDE data/continuum_reddening_flat.dat

END
