import numpy as np
from math import comb

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer

def calc_pool_conc(x, data_arr, bools, alpha):
    """Calculate diversification of a given pool. Used to 
    find the best pool.

    x : bool
        Countries to consider in the pool
    data_arr : np.array
        Numpy array with annual damages for all countries
    bools : np.array
        Numpy array with the same shape as data, indicating when 
        annual damages are higher/lower than the country VaR
    alpha : float
        Point at which to calculate VaR and ES
    """

    dam = data_arr[:,x]
    cntry_bools = bools[:,x]
    tot_damage = dam.sum(1)
    
    VAR_tot = np.quantile(tot_damage[~np.isnan(tot_damage)], alpha)
    bool_tot = tot_damage >= VAR_tot

    ES_cntry = []
    MES = []

    for cntry_pos in range(dam.shape[1]):
        dummy_dam = dam[:,cntry_pos][cntry_bools[:,cntry_pos]]

        ES_cntry.append(np.nanmean(dummy_dam))
        MES.append(np.nanmean(dam[:,cntry_pos][bool_tot]))

    ES_cntry = np.array(ES_cntry)
    MES = np.array(MES)

    # if no countries are picked
    if x.sum() == 0:
        POOL_CONC = 1.
        IND_CONC = 1.
    else:
        IND_CONC = MES / ES_cntry
        ES_tot = np.nansum(MES)
        POOL_CONC = ES_tot / np.nansum(ES_cntry)

    return np.round(POOL_CONC, 2) #, IND_CONC, MES, ES_cntry, tot_damage

def calc_pools_conc(x, data, bools, alpha, N, fixed_pools=None):
    """Calculate diversification of N pools where all passed countries
    must be in one pool

    x : np.array
        Integers. Integers assess what pool do countries join. 
        It must hold that x.size equals + fixed_pools.size equals n_countries.
    data : np.array
        Numpy array with annual damages for all countries
    bools : np.array
        Numpy array with the same shape as data, indicating when 
        annual damages are higher/lower than the country VaR
    alpha : float
        Confidence level to estimating VaR and ES
    N : int
        Number of pools
    fixed_pools : np.array
        Integers for countries which will always join the same pool. It 
        concatenates x from the left (beginning of the array). It must hold that 
        x.size equals + fixed_pools.size equals n_countries.
    """

    CONC_POOL = []

    if fixed_pools is not None:
        x = np.hstack([fixed_pools, x])

    for i in range(1, N+1):
        countries_in_pool = x == i
        conc_pool = calc_pool_conc(countries_in_pool, data, bools, alpha)[0]

        CONC_POOL.append(conc_pool)

    return np.array([CONC_POOL])


class PoolOptimizationProblem(ElementwiseProblem):
    def __init__(self, nominals, max_nominal, data, bools, alpha, N, fun, **kwargs):
        self.data_arr = data
        self.bools = bools
        self.alpha = alpha
        self.N = N
        self.fun = fun
        self.nominals = np.array(nominals)
        self.n_countries = len(nominals)
        self.max_nominal = max_nominal
        super().__init__(
            n_var=self.data_arr.shape[1],
            n_obj=1,  
            n_constr = 1,
            xl=0,                  
            xu=self.N - 1,
            type_var=int,
            vars=[Integer(0, self.n_countries - 1) for _ in range(self.n_countries)],
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pools = {i: [] for i in np.unique(x)}
        for i, pool_id in enumerate(x):
            if len(np.where(x == i)[0]) > 0:
                pool_mask = np.where(x == i)[0]
                pools[i].append(pool_mask)
        
        total_concentration = 0
        for pool_key, pool_countries in pools.items():
            pool1_col = self.data_arr.columns[pool_countries[0]]
            pool1_data = self.data_arr[pool1_col].values
            pool1_bools = self.bools[pool1_col].values
            conc = self.fun(np.arange(0, len(pool_countries[0])), pool1_data, pool1_bools, self.alpha)
            total_concentration += conc
        constraints = 0
        for members in pools.values():
            pool_nominal_diff = np.sum(self.nominals[members[0]]) - self.max_nominal
            if pool_nominal_diff > 0:
                constraints += pool_nominal_diff

        out["F"] = total_concentration/len(pools)
        out["G"] = constraints

def pop_num(n, k):
    combinations = comb(n + k - 1, k)
    return combinations