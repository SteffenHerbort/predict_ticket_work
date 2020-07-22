# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:21:59 2020

@author: Steffen Herbort
"""

import parse as prs
import datetime
import itertools


def str2date( string_representing_date ):
    """Converts strings of dates into 'datetime' objects

    Expected date fromat is {}-{}-{}T{}:{}:{}.{}+{}

    Parameters
    ----------
    string_representing_date : str, optional
        The string containing a date
        
    Returns
    -------
    'datetime' object
    
    """

    out = prs.parse("{}-{}-{}T{}:{}:{}.{}+{}", string_representing_date)
    #this removes the timezone infos, but they're not needed anyway
    return datetime.datetime(year = int(out[0]), 
                             month= int(out[1]), 
                             day=   int(out[2]), 
                             hour=  int(out[3]), 
                             minute=int(out[4]), 
                             second=int(out[5]))
 

def get_parameter_combinations( param_dict ):
    """creates all parameter combinations from a dictionary

    Example:
        input: dictionary
               d = {'param1' : [1,2], 'param2' : ['a', 'b']}
        
        output: list of dictionaries
            [ {'param1' : 1, 'param2' : 'a'},
              {'param1' : 1, 'param2' : 'b'},
              {'param1' : 2, 'param2' : 'a'},
              {'param1' : 2, 'param2' : 'b'}
            ]

    Parameters
    ----------
    param_dict : dictionary
        A dictionary containing valid values for all parameters
        e.g.
        d = {'param1' : [1,2], 'param2' : ['a', 'b']}
        
    Returns
    -------
    list of dictionaries that contain parameter sets
    
    """

    
    data = []
    for key in param_dict.keys():
        data.append( param_dict[key] )
        
    combinations = list(itertools.product(*data))
    
    out = []
    for elem in combinations:
        tmp_set_of_params = {}
        for idx, key in enumerate(param_dict.keys()):
            tmp_set_of_params[key] = elem[idx]
        out.append(tmp_set_of_params)
    
    return out