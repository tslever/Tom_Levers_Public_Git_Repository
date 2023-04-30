import pkg_resources
from subprocess import call

def upgrade_all_pip_packages():
    packages = [dist.project_name for dist in pkg_resources.working_set]
    call("pip install --upgrade " + ' '.join(packages), shell=True)

if __name__ == "__main__":
    upgrade_all_pip_packages()