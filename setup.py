from setuptools import setup

# with open('requirements.txt') as requirements:
#     reqs = requirements.read().splitlines()

setup(
    name='cf_tools',
    version='0.1',
    description=('A library for collaborative filtering'
                 'and similarity-based recommendations'),
    author='Matteo Banerje',
    author_email='matteobanerjee@gmail.com',
    url='https://github.com/matteobanerjee/collaborative_filtering_tools',
    # install_requires=reqs, <-- Currently can't get numpy to install this way
    packages=['cf_tools', 'cf_tools/core', 'cf_tools/recommender'],
    license='MIT'
)
