###############################################################################
### Loop through all the views and hand pose and perform model fitting      ###
###############################################################################

import os

def run(command):
	print(command)
	os.system(command)

for view in range(1,6):
	run('python utils_model_fitting.py --view ' + str(view) + ' --file fist_00 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file fist_03 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file hook_00 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file hook_03 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_mcp_00 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_mcp_03 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_ip_00 --mode auto')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_ip_03 --mode auto')


for view in range(1,6):
	run('python utils_model_fitting.py --view ' + str(view) + ' --file fist_00 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file fist_03 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file hook_00 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file hook_03 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_mcp_00 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_mcp_03 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_ip_00 --mode manual')
	run('python utils_model_fitting.py --view ' + str(view) + ' --file thumb_ip_03 --mode manual')	


print('Done')