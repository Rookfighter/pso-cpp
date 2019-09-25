import subprocess
import os
import sys

def run_doxygen(folder):
    """Run the doxygen make command in the designated folder"""
    try:
        retcode = subprocess.call("cd %s; make" % folder, shell=True)
        if retcode < 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)


extensions = ['sphinx.ext.pngmath', 'sphinx.ext.todo', 'breathe']

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

if read_the_docs_build:
    run_doxygen('doxygen')

breathe_projects = {"doxygen": "xml"}
breathe_default_project = "doxygen"
