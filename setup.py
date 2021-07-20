from setuptools import setup, find_packages


if __name__ == '__main__':
    packages = find_packages(
        exclude=(
            'deprecated', 'deprecated.*', 'scripts', 'scripts.*',
            'unittest', 'unittest.*'
        ),
        include=('mycv')
    )

    setup(
        name='mycv',
        version='0.1',
        description='A personal computer vision repository using pytorch',
        author='Zhihao Duan',
        author_email='duan90@purdue.edu',
        url='https://github.com/duanzhiihao/mycv',
        packages=packages
    )
