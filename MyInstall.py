import sys
import subprocess

# implement pip as a subprocess:
pacages = ["pandas==1.3.4","setuptools==58.0.4", "networkx~=2.6.3", "fire==0.4.0", "numpy==1.20.0", "numba==0.54.1",
           "scikit-learn==1.0.1", "Pillow==8.4.0","tqdm==4.62.3", "future==0.18.2", "colorama==0.4.4", "matplotlib==3.5.0", "tensorboard==2.7.0", "node2vec"]
def install():
    global reqs
    string_to_print = "pytorch  ,torchvision  ,torchaudio  ,cudatoolkit=11.3"	
    sign=str((len(string_to_print)+2)*'=') 
    print('\n'+sign+'\n'+string_to_print+'\n'+sign+'\n')	
    subprocess.check_call([sys.executable, '-m', 'conda', 'install',
    'pytorch','torchvision','torchaudio','cudatoolkit=11.3','-c','pytorch','-c','conda-forge'])
    for i in pacages:
        sign=str((len(i)+2)*'=')
        print('\n'+sign+'\n'+i+'\n'+sign+'\n')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               i])


    
    # process output with an API in the subprocess module:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip',
                                                'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    print(installed_packages)

if __name__ == '__main__':
    install()
