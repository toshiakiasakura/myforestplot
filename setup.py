from setuptools import setup
from codecs import open
from os import path

current = path.abspath(path.dirname(__file__))
INSTALL_REQUIRES = [
    'matplotlib>=3.5.1',
]
EXTRAS_REQUIRE = {}

# long_description(後述)に、GitHub用のREADME.mdを指定
with open(path.join(current, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='myforestplot', # パッケージ名(プロジェクト名)
    packages=['myforestplot'], # パッケージ内(プロジェクト内)のパッケージ名をリスト形式で指定
    license='MIT', # ライセンス

    author='toshiakiasakura', # パッケージ作者の名前
    author_email='wordpress.akitoshi@gmail.com', # パッケージ作者の連絡先メールアドレス
    url='https://toshiakiasakura.github.io/myforestplot', # パッケージに関連するサイトのURL(GitHubなど)

    description='Create forestplot by Python.', # パッケージの簡単な説明
    long_description=long_description, # PyPIに'Project description'として表示されるパッケージの説明文
    long_description_content_type='text/markdown', # long_descriptionの形式を'text/plain', 'text/x-rst', 'text/markdown'のいずれかから指定
    keywords='forestplot odds-ratio relative-risk meta-analysis', # PyPIでの検索用キーワードをスペース区切りで指定

    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Framework :: Matplotlib',
    ], # パッケージ(プロジェクト)の分類。https://pypi.org/classifiers/に掲載されているものを指定可能。
)

