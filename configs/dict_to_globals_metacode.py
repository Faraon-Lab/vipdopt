### For each key in the yaml, produces a line that says 
### key = parameters.get('key')

import yaml
import pyperclip

#* Load YAML file
yaml_filename = 'configs/test_config_sony.yaml'
with open( yaml_filename, 'rb' ) as yaml_file:
	parameters = yaml.safe_load( yaml_file )

dict_name = 'param_dict'
output_txt = ''
for key in list(parameters.keys()):
    # output_txt += f'{key} = {dict_name}.get("{key}")\n' # DEPRECATED
    output_txt += f'{key} = get_check_none({dict_name}, "{key}")\n'

print(output_txt)
pyperclip.copy(output_txt)
print('Text copied to clipboard.')