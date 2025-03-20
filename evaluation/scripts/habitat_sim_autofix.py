import site
import glob
import os

site_pkgs = site.getsitepackages()
habitat_sim_paths = [
    path for pkg in site_pkgs for path in glob.glob(os.path.join(pkg, 'habitat_sim-0.1.7*.egg', 'habitat_sim', 'utils', 'common.py'))
]

if habitat_sim_paths:
    file_path = habitat_sim_paths[0]
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace('np.float', 'np.float32')
    with open(file_path, 'w') as f:
        f.write(content)
    print(f'Fixed outdated NumPy type in: {file_path}')
else:
    print('Error: habitat_sim/utils/common.py not found!')
