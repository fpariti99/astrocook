from .cookbook_absorbers_old import CookbookAbsorbersOld
from .functions import resol_check, to_z, trans_parse
from .line_list import LineList
from .vars import resol_def, xem_d

import logging
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as au
from scipy.signal import find_peaks

class CookbookAbsorbers(CookbookAbsorbersOld):

    def __init__(self):
        super(CookbookAbsorbers, self).__init__()


    def find_lines(self, kind='abs', prominence=None, append=True):
        """ @brief Find lines
        @details Find absorption or emission lines, based on their prominence.
        @url absorbers_cb.html#find-lines
        @param kind Kind
        @param prominence Prominence
        @param append Append to existing line list
        @return 0
        """

        try:
            kind = str(kind)
            prominence = None if prominence in [None, 'None'] \
                else float(prominence)
            append = str(append) == 'True'
        except:
            logging.error(msg_param_fail)
            return 0

        if kind not in ['abs', 'em']:
            logging.error("`kind` should be `abs` or `em`. Aborting.")
            return 0

        spec = self.sess.spec
        fact = -1 if kind=='abs' else 1

        ynorm = fact*(spec._t['y'])
        if prominence is None: prominence = 5*(spec._t['dy'])

        peaks, properties = find_peaks(ynorm, prominence=prominence)
        lines = LineList(row=spec._t[peaks], source='y', kind=kind,
                         xunit=spec._xunit, yunit=spec._yunit, meta=spec._meta)
        lines.append_replace(append, self.sess)

        return 0


    def model_lya(self):
        """@brief Model Ly-a forest 🚧
        @details 🚧
        @url absorbers_cb.html#model-ly-a-forest
        """

        return 0




    def compute_ew(self, x1, x2, rel_err_rmse, rel_err_mad):
        try:
           x1 = float(x1)
           x2 = float(x2)
           

           if x1 >= x2:
             raise ValueError("x1 must be less than x2.")
        except Exception as e:
             logging.error(f"Error in EW calculation: {str(e)}")

             return None, None, None, None, None  # Ritorna valori nulli in caso di errore

        unit = self.sess.spec._t['x'].unit
        sel = np.logical_and(np.array(self.sess.spec._t['x']) > x1,
                         np.array(self.sess.spec._t['x']) < x2)

        

        t = self.sess.spec._t[sel]
 
        if np.any(np.isnan(t['y'])):
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
        dx = (t['xmax'] - t['xmin']).to(unit)
    
        ew = np.nansum(dx * (1 - np.array(t['y'] / t['cont'])))  # Calcolo EW
    
        # Calcola gli errori del continuo
        cont_error_rmse = rel_err_rmse * t['cont']
        cont_error_mad = rel_err_mad * t['cont']
    
        # Calcola dEW usando RMSE
        dEW_rmse = np.nansum((dx * (-1 / t['cont']))**2 * (t['dy']**2 + (cont_error_rmse)**2))
        dEW_rmse = np.sqrt(dEW_rmse)
    
        # Calcola dEW usando MAD
        dEW_mad = np.nansum((dx * (-1 / t['cont']))**2 * (t['dy']**2 + (cont_error_mad)**2))
        dEW_mad = np.sqrt(dEW_mad)
    
        

        #print(f"DEBUG - Types: EW={type(ew)}, dEW_rmse={type(dEW_rmse)}, dEW_mad={type(dEW_mad)}")
        #print(f"DEBUG - Values: dEW_rmse={dEW_rmse}, dEW_mad={dEW_mad}")


        # Calcola sigma per entrambi i casi
        sigma_rmse = ew / dEW_rmse if dEW_rmse != 0 else np.nan
        sigma_mad = ew / dEW_mad if dEW_mad != 0 else np.nan
 
        
        return ew, dEW_rmse, sigma_rmse, dEW_mad, sigma_mad

    

        
    
    def ew_manual(self,row_name, x1,x2):
        """@brief Compute Equivalent Width from input interval
        @details Computes the Equivalent Width for the specified range of wavelengths.
        @param row_name: Insert transition name
        @param x1: Starting wavelength of the interval
        @param x2: Ending wavelength of the interval
        @return The computed EW value or logs an error if inputs are invalid.
        """
        
        x1 = float(x1)
        x2 = float(x2)
        
        ew, dEW, sigma = self.compute_ew(x1, x2)
        if ew is not None: 
           unit = self.sess.spec._t['x'].unit
           logging.info(f"EW for line {row_name} between {x1:.4f} and {x2:.4f} {unit}: {ew:.3e} ± {dEW:.3e} {unit}")

    
        return 0



    def ew_table(self, table_path=None):
        """@brief Compute EW from table
        @details Compute EW from table using both RMSE and MAD for continuum error estimation.
        @param table_path: Path to the table containing intervals (optional)
        """
        unit = self.sess.spec._t['x'].unit
        try:
           try:
              data = pd.read_csv(table_path, sep=';')
              if data.shape[1] == 1:
                  raise ValueError("File letto con una sola colonna, riprovo con ','")
              print("File caricato con separatore ';'.")
           except Exception as e:
                logging.warning(f"Errore con ';': {e}, riprovo con ','...")
                try:
                   data = pd.read_csv(table_path, sep=',')
                   if data.shape[1] == 1:
                      raise ValueError("Il file sembra ancora errato, controlla il formato.")
                   print("File caricato con separatore ','.")
                except Exception as e:
                    logging.error(f"Errore nel leggere il file CSV: {e}")
                    return None

           if data.empty:
               logging.error("Il file CSV è vuoto.")
               return None

           mode = input("Enter mode (e.g., high or low): ")
           ew_col_name = f"ew {mode}"
           dew_rmse_col_name = f"dew_rmse {mode}"
           dew_mad_col_name = f"dew_mad {mode}"
           sigma_rmse_col_name = f"sigma_rmse {mode}"
           sigma_mad_col_name = f"sigma_mad {mode}"

           data[ew_col_name] = None
           data[dew_rmse_col_name] = None
           data[dew_mad_col_name] = None
           data[sigma_rmse_col_name] = None
           data[sigma_mad_col_name] = None

           if unit == au.nm:

              for index, row in data.iterrows():
                  row_name = row['Transition']
                  x1 = row['Observed_Lambda_Min']* au.nm
                  x2 = row['Observed_Lambda_Max']* au.nm

                  rel_err_rmse = row['Relative_Error_rmse']
                  rel_err_mad = row['Relative_Error_mad']

                  

                  # 🔴 Stampa i tipi delle variabili per debugging
                  #print(f"Row {index}: x1={x1} ({type(x1)}), x2={x2} ({type(x2)}), rel_err_rmse={rel_err_rmse} ({type(rel_err_rmse)}), rel_err_mad={rel_err_mad} ({type(rel_err_mad)}) ")

                 
 
                  
                  ew, dEW_rmse, sigma_rmse, dEW_mad, sigma_mad = self.compute_ew(
    x1.value, x2.value, rel_err_rmse, rel_err_mad
)     
                  # 🔴 Stampa i tipi prima di scriverli nel DataFrame
                  #print(f"Row {index}: EW={ew} ({type(ew)}), dEW_rmse={dEW_rmse} ({type(dEW_rmse)})")


                  if ew is not None:
                      # 🔴 Stampa i tipi prima di scriverli nel DataFrame
                      #print(f"Row {index}: EW={ew} ({type(ew)}), dEW_rmse={dEW_rmse} ({type(dEW_rmse)}), dEW_mad={dEW_mad} ({type(dEW_mad)}), sigma_rmse={sigma_rmse} ({type(sigma_rmse)}), sigma_mad={sigma_mad} ({type(sigma_mad)})")
                      #print(f"Row {index}: dEW_rmse={dEW_rmse}, unit={dEW_rmse.unit}, dEW_mad={dEW_mad}, unit={dEW_mad.unit}")


                      data.at[index, ew_col_name] = ew.value if not np.isnan(ew) else np.nan
                      data.at[index, dew_rmse_col_name] = dEW_rmse.value if not np.isnan(dEW_rmse) else np.nan
                      data.at[index, dew_mad_col_name] = dEW_mad.value if not np.isnan(dEW_mad) else np.nan
                      data.at[index, sigma_rmse_col_name] = sigma_rmse.value if not np.isnan(sigma_rmse) else np.nan
                      data.at[index, sigma_mad_col_name] = sigma_mad.value if not np.isnan(sigma_mad) else np.nan


           elif unit == au.angstrom:

              for index, row in data.iterrows():
                  row_name = row['Transition']
                  x1 = row['Observed_Lambda_Min'] * unit.nm
                  x2 = row['Observed_Lambda_Max'] * unit.nm
                  rel_err_rmse = row['Relative_Error_rmse']
                  rel_err_mad = row['Relative_Error_mad']
 
                 

                  ew, dEW_rmse, sigma_rmse, dEW_mad, sigma_mad = self.compute_ew(
    x1.value, x2.value, rel_err_rmse, rel_err_mad
)

                  if ew is not None:
                    
                     data.at[index, ew_col_name] = ew.value
                     data.at[index, dew_rmse_col_name] = dEW_rmse.value
                     data.at[index, dew_mad_col_name] = dEW_mad.value
                     data.at[index, sigma_rmse_col_name] = sigma_rmse.value
                     data.at[index, sigma_mad_col_name] = sigma_mad.value

           else:
                 logging.error(f"Unsupported unit for wavelength: {unit}")
                 return 0

           data.to_csv(table_path, index=False)
           print(f"Results saved to {table_path}")

        except Exception as e:
             logging.error(f"Error loading the table: {e}")
             return 0

        return 1
    
   


    def model_metals(self, series, zem, no_ly=True, use_lines=False):
        """@brief Model metals
        @details Model metal absorbers, based on transition and emission
        redshift.
        @url absorbers_cb.html#model-metals
        @param series Transitions
        @param zem Emission redshift
        @param no_ly Exclude Lyman forest
        @param use_lines Use line list to define absorbers
        @url absorbers_cb.html#model-metals
        """

        try:
            zem = float(zem)
            no_ly = str(no_ly) == 'True'
            use_lines = str(use_lines) == 'True'
        except:
            logging.error(msg_param_fail)
            return 0

        if no_ly:
            trans = trans_parse(series)
            z_start = to_z(xem_d['Ly_a']*(1+zem), trans[0])
        else:
            z_start = 0
        z_end = zem

        check, resol = resol_check(self.sess.spec)
        if resol is None:
            self.sess.spec._resol_est(3, True)
            resol = self.sess.spec._t['resol'][0]

        col = 'deabs' if 'deabs' in self.sess.spec._t.colnames else 'y'

        if use_lines and self.sess.lines is None:
            logging.warning("I didn't find a line list. Ignoring it.")
            use_lines = False

        if use_lines:
            self.systs_new_from_lines(series=series, z_start=z_start,
                                      z_end=z_end, refit_n=0, append=True)
        else:
            self.systs_new_from_like(series=series, col=col, z_start=z_start,
                                     z_end=z_end, modul=10, sigma=3,
                                     distance=3, resol=resol, append=True)
        self.systs_fit(refit_n=0)
        return 0


    def identify_unknown(self, x, x_doublet="", dz_systs=1e-4, dz_doublet=1e-5,
                       dz_unknown=1e-5, dz_galact=1e-2, z_min=0, z_max=9):
        """ @brief Identify unknown lines
        @details Complete system identification by (1) associating unknown lines
        to known systems; (2) identifying doublets from their wavelength ratio;
        (3) identifying unknown lines by finding possible redshift coincidences;
        and (4) identifying possible galactic lines. Identification is done
        using a reference list of transitions) 🚧
        @url absorbers_cb.html#identify-unknown-lines
        @param x List of line wavelengths (nm)
        @param x_doublet Test wavelengths of the doublets (nm; e.g. 500,501;600-601)
        @param dz_systs Threshold for redshift coincidence with known systems
        @param dz_doublet Threshold for redshift coincidence between doublet members
        @param dz_unknown Threshold for redshift coincidence between unknown lines
        @param dz_galact Threshold for coincidence with redshift 0
        @param z_min Minimum redshift
        @param z_max Maximum redshift
        @return 0
        """

        try:
            x = np.array(x[1:-1].split(','), dtype=float) if type(x)==str \
                else np.array(x)
            dz_systs = float(dz_systs)
            dz_doublet = float(dz_doublet)
            dz_unknown = float(dz_unknown)
            dz_galact = float(dz_galact)
            z_min = float(z_min)
            z_max = float(z_max)
        except:
            logging.error(msg_param_fail)
            return 0

        self._systs_assoc(x, dz_systs)
        if x_doublet not in [None, ""]: self._doublet_iden(x_doublet, dz_doublet)
        self._unknown_iden(x, dz_unknown, z_min, z_max)
        self._galact_iden(x, dz_galact)

        return 0



    def check_systs(self):
        """@brief Check system list 🚧
        @details 🚧
        @url absorbers_cb.html#check-system-list
        """

        return 0
