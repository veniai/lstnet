# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/lstnet/gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['lstnet.io.emissivity', 'lstnet.io.modis', 'lstnet.io.aster_ged', 'lstnet.io.surfrad', 'lstnet.io.pku', 'lstnet.io.hiwater', 'lstnet.io.raster', 'lstnet.validation', 'lstnet.stats', 'lstnet.plotting', 'lstnet.mcp_server', 'lstnet.dayornight', 'lstnet.qc', 'lstnet.ground_lst', 'lstnet.config', 'lstnet.models', 'lstnet.sites'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='lstnet-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lstnet-gui',
)
