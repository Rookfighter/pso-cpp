import subprocess
import os
import sys

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
extensions = ['breathe']
breathe_projects = {'doxygen': 'xml'}
breathe_default_project = 'doxygen'

def run_doxygen(folder):
    """Run the doxygen command in the designated folder"""
    try:
        retcode = subprocess.call('cd %s; doxygen' % folder, shell=True)
        if retcode < 0:
            sys.stderr.write('doxygen terminated by signal %s' % (-retcode))
    except OSError as e:
        sys.stderr.write('doxygen execution failed: %s' % e)

def generate_doxygen_xml(app):
    if read_the_docs_build:
        run_doxygen('.')

def setup(app):
    app.connect('builder-inited', generate_doxygen_xml)
