from own_package.combination import decomp_combi

def selector(case, **kwargs):
    if case == 1:
        var_name = kwargs['var_name']
        numel = kwargs['numel']
        subgroup_size = kwargs['subgroup_size']
        decomp_combi(var_name=var_name, numel=numel, subgroup_size=subgroup_size)


selector(1, var_name='IND', numel=100, subgroup_size=30)
