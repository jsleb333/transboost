import os
from setuptools import setup, find_packages

current_file_path = os.path.abspath(os.path.dirname(__file__))

readme_file_path = os.path.join(current_file_path, 'README.md')
with open(readme_file_path, 'r') as f:
    readme = f.read()

version_file_path = os.path.join(current_file_path, 'version.py')
with open(version_file_path, 'rb') as f:
    # pylint: disable=exec-used,undefined-variable
    exec(compile(f.read(), version_file_path, 'exec'), globals(), locals())
    version = __version__

packages = find_packages()
setup(
    name='TransBoost',
    version=version,
    author='Jean-Samuel Leboeuf and Léandre Gagnon-Lewis',
    author_email='jean-samuel.leboeuf.1@ulaval.ca',
    download_url='https://github.com/jsleb333/transboost.zip',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=packages,
    extras_require={'Visualization': ['matplotlib'],
                    'Traceback Pickling': ['tblib']},
    python_requires='>=3.6',
    description='Multiscale Boosting',
    long_description=readme,
    long_description_content_type='text/markdown',
)
