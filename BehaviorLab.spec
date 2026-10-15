# Build only from the lean environment: python -m PyInstaller BehaviorLab.spec
from pathlib import Path
from behaviorlab import __version__
project = Path(SPECPATH)
a = Analysis([str(project / 'desktop_app.py')], pathex=[str(project)], binaries=[], datas=[],
             hiddenimports=['PIL._tkinter_finder'],
             excludes=['torch', 'torchvision', 'ultralytics', 'gradio', 'scipy', 'matplotlib',
                       'pandas', 'polars', 'onnx', 'onnxruntime', 'IPython', 'pytest'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='BehaviorLab',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
collect = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=f'BehaviorLab-{__version__}')
