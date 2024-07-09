import os
import json
import numpy as np

from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

quantify_shifts_dir = config.output_dir + 'shift_quantities/'
setup_dirs          = os.listdir(quantify_shifts_dir)

covariate_shift_str   = ''
conditional_shift_str = ''
covariate_shift_rel_to_final_str   = ''
conditional_shift_rel_to_final_str = ''

for setup_dir in setup_dirs:
    setup_dir_path = quantify_shifts_dir + setup_dir
    filename       = setup_dir_path + '/seed1007/quantify_shift_Wasserstein_dists.json'
    if not os.path.exists(filename):
        continue
    with open(filename, 'r') as f:
        w_dists = json.load(f)
    covariate_shift_ratios = np.array(w_dists['Covariate shift W2s'])/w_dists['Source covariate shift W2']
    covariate_shift_str += setup_dir + ' & ' + ' & '.join(map(str, covariate_shift_ratios.tolist())) + ' \\\\\n'
    
    conditional_shift_ratios = np.array(w_dists['Conditional shift W2s'])/w_dists['Source conditional shift W2']
    conditional_shift_str += setup_dir + ' & ' + ' & '.join(map(str, conditional_shift_ratios.tolist())) + ' \\\\\n'

    rel_cov_shift_key = 'Covariate shift relative to final test W2s'
    covariate_shift_rel_to_final_ratios = np.array(w_dists[rel_cov_shift_key][:-1])/w_dists[rel_cov_shift_key][-1]
    covariate_shift_rel_to_final_str += setup_dir + ' & ' + ' & '.join(map(str, covariate_shift_rel_to_final_ratios.tolist())) + ' \\\\\n'

    rel_cond_shift_key = 'Conditional shift relative to final test W2s'
    conditional_shift_rel_to_final_ratios = np.array(w_dists[rel_cond_shift_key][:-1])/w_dists[rel_cond_shift_key][-1]
    conditional_shift_rel_to_final_str += setup_dir + ' & ' + ' & '.join(map(str, conditional_shift_rel_to_final_ratios.tolist())) + ' \\\\\n'
    
with open(quantify_shifts_dir + 'covariate_shift_ratios_table.txt', 'w') as f:
    f.write(covariate_shift_str)
    
with open(quantify_shifts_dir + 'conditional_shift_ratios_table.txt', 'w') as f:
    f.write(conditional_shift_str)

with open(quantify_shifts_dir + 'covariate_shift_relative_to_final_ratios_table.txt', 'w') as f:
    f.write(covariate_shift_rel_to_final_str)

with open(quantify_shifts_dir + 'conditional_shift_relative_to_final_ratios_table.txt', 'w') as f:
    f.write(conditional_shift_rel_to_final_str)