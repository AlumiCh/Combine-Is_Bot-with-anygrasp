#13 19.77 Building wheels for collected packages: PyOpenGL
#13 19.77   Building wheel for PyOpenGL (pyproject.toml): started
#13 20.91   Building wheel for PyOpenGL (pyproject.toml): finished with status 'done'
#13 20.91   Created wheel for PyOpenGL: filename=pyopengl-3.1.0-py3-none-any.whl size=1745256 sha256=bae0ba4386fa085d0c9341234ce3663a758b296eb20e22e5992316a848ea3b07
#13 20.91   Stored in directory: /tmp/pip-ephem-wheel-cache-g0orng8u/wheels/a1/3c/d2/1f9533f908d86176637521e533c6cdb2d4e48b59003b5c3f19
#13 20.92 Successfully built PyOpenGL
#13 21.25 Installing collected packages: PyOpenGL, pyglet, numpy, freetype-py, pyrender
#13 21.25   Attempting uninstall: PyOpenGL
#13 21.25     Found existing installation: PyOpenGL 3.1.10
#13 21.49     Uninstalling PyOpenGL-3.1.10:
#13 23.62       Successfully uninstalled PyOpenGL-3.1.10
#13 25.13   Attempting uninstall: numpy
#13 25.13     Found existing installation: numpy 2.2.6
#13 25.18     Uninstalling numpy-2.2.6:
#13 25.60       Successfully uninstalled numpy-2.2.6
#13 26.89 
#13 26.92 ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#13 26.92 dm-control 1.0.34 requires pyopengl>=3.1.4, but you have pyopengl 3.1.0 which is incompatible.
#13 26.92 numba 0.57.1+1.gf851d279c requires numpy<1.25,>=1.21, but you have numpy 1.26.4 which is incompatible.
#13 26.92 opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
#13 26.92 torchtext 0.16.0a0 requires torch==2.1.0a0+b5021ba, but you have torch 2.1.0 which is incompatible.
#13 26.92 Successfully installed PyOpenGL-3.1.0 freetype-py-2.5.1 numpy-1.26.4 pyglet-2.1.6 pyrender-0.1.45
#13 26.93 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
#13 29.83 Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
#13 33.53 Collecting PyOpenGL==3.1.5
#13 34.12   Downloading PyOpenGL-3.1.5-py3-none-any.whl.metadata (3.2 kB)
#13 34.35 Downloading PyOpenGL-3.1.5-py3-none-any.whl (2.4 MB)
#13 35.32    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.4/2.4 MB 3.3 MB/s  0:00:00
#13 35.68 Installing collected packages: PyOpenGL
#13 35.68   Attempting uninstall: PyOpenGL
#13 35.68     Found existing installation: PyOpenGL 3.1.0
#13 35.84     Uninstalling PyOpenGL-3.1.0:
#13 35.86       Successfully uninstalled PyOpenGL-3.1.0
#13 37.31 ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#13 37.31 pyrender 0.1.45 requires PyOpenGL==3.1.0, but you have pyopengl 3.1.5 which is incompatible.
#13 37.31 Successfully installed PyOpenGL-3.1.5
#13 37.31 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
#13 DONE 39.4s

#14 [10/30] COPY pointnet2_ops pointnet2_ops
#14 DONE 0.0s

#15 [11/30] RUN pip install ./pointnet2_ops
#15 0.343 Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
#15 0.343 Processing ./pointnet2_ops
#15 0.345   Installing build dependencies: started
#15 5.755   Installing build dependencies: finished with status 'done'
#15 5.755   Getting requirements to build wheel: started
#15 5.912   Getting requirements to build wheel: finished with status 'error'
#15 5.916   error: subprocess-exited-with-error
#15 5.916   
#15 5.916   × Getting requirements to build wheel did not run successfully.
#15 5.916   │ exit code: 1
#15 5.916   ╰─> [19 lines of output]
#15 5.916       /tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
#15 5.916         warnings.warn(
#15 5.916       Traceback (most recent call last):
#15 5.916         File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
#15 5.916           main()
#15 5.916         File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
#15 5.916           json_out["return_val"] = hook(**hook_input["kwargs"])
#15 5.916         File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 143, in get_requires_for_build_wheel
#15 5.916           return hook(config_settings)
#15 5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 331, in get_requires_for_build_wheel
#15 5.916           return self._get_build_requires(config_settings, requirements=[])
#15 5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 301, in _get_build_requires
#15 5.916           self.run_setup()
#15 5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 512, in run_setup
#15 5.916           super().run_setup(setup_script=setup_script)
#15 5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 317, in run_setup
#15 5.916           exec(code, locals())
#15 5.916         File "<string>", line 5, in <module>
#15 5.916       ModuleNotFoundError: No module named 'torch'
#15 5.916       [end of output]
#15 5.916   
#15 5.916   note: This error originates from a subprocess, and is likely not a problem with pip.
#15 8.573 ERROR: Failed to build 'file:///workspace/pointnet2_ops' when getting requirements to build wheel
#15 ERROR: process "/bin/sh -c pip install ./pointnet2_ops" did not complete successfully: exit code: 1
------
 > [11/30] RUN pip install ./pointnet2_ops:
5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 512, in run_setup
5.916           super().run_setup(setup_script=setup_script)
5.916         File "/tmp/pip-build-env-__fjrfqs/overlay/local/lib/python3.10/dist-packages/setuptools/build_meta.py", line 317, in run_setup
5.916           exec(code, locals())
5.916         File "<string>", line 5, in <module>
5.916       ModuleNotFoundError: No module named 'torch'
5.916       [end of output]
5.916   
5.916   note: This error originates from a subprocess, and is likely not a problem with pip.
8.573 ERROR: Failed to build 'file:///workspace/pointnet2_ops' when getting requirements to build wheel
------
graspgen_cuda121.dockerfile:27
--------------------
  25 |     # Install pointnet2 modules
  26 |     COPY pointnet2_ops pointnet2_ops
  27 | >>> RUN pip install ./pointnet2_ops
  28 |     
  29 |     # Diffusion dependencies
--------------------
ERROR: failed to solve: process "/bin/sh -c pip install ./pointnet2_ops" did not complete successfully: exit code: 1



https://msub.xn--m7r52rosihxm.com/api/v1/client/subscribe?token=36b1d43a3543a98b0c337d3fd0c09a83