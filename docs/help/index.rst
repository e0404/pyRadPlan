.. _help:

Help and support
================

If you run into problems, have questions, or want to suggest improvements, the project's GitHub repository is the best place to start.

Source code
-----------

The full source code lives at `github.com/e0404/pyRadPlan
<https://github.com/e0404/pyRadPlan>`_.

Reporting issues
----------------

If you find a bug or would like to request a feature, please open an issue on the
`issue tracker <https://github.com/e0404/pyRadPlan/issues>`_. Issue templates are provided
for bug reports and feature requests to help you include the relevant details.

Frequently asked questions
--------------------------

.. _faq-gui-windows-msvc:

The GUI does not start on Windows, or says PySide6 is not installed although it is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affects:** pyRadPlan 0.9.0 and earlier on Windows, mostly older machines.

**Symptoms**

* After ``pip install "pyRadPlan[gui]"``, starting the GUI fails with *"pyRadPlan GUI requires
  the 'gui' optional dependencies (PySide6, pyqtgraph)"*, although both are installed.
  Reinstalling them does not reliably help.
* The standalone Windows installer finishes, but pyRadPlan does not open and shows no error.

**Cause**

PySide6 ships the Microsoft C++ runtime (``msvcp140.dll``) that Qt needs. SimpleITK uses the copy
installed in Windows instead, and pyRadPlan imported SimpleITK before Qt. Windows only loads a DLL
once per process, so Qt ran on the copy installed in Windows. If that copy comes from an outdated
Microsoft Visual C++ Redistributable (older than version 14.40), Qt fails to load. pyRadPlan
reported this failure as missing GUI dependencies.

**Solution**

Install the latest **Microsoft Visual C++ Redistributable (x64)** from the
`Microsoft download page <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_,
then start pyRadPlan again. This fixes both the ``pip`` installation and the standalone app.

To check whether your machine is affected, run in PowerShell:

.. code-block:: powershell

   # Version of the runtime installed in Windows; affected if below 14.40
   (Get-Item C:\Windows\System32\msvcp140.dll).VersionInfo.FileVersion

   # Works on affected machines ...
   python -c "from PySide6 import QtCore; print('ok')"
   # ... while this fails or exits without output
   python -c "import SimpleITK; from PySide6 import QtCore; print('ok')"

**Fixed in:** releases after 0.9.0. The standalone app ships its own up-to-date runtime and no
longer depends on the one installed in Windows. The ``pyRadPlanGUI`` command loads Qt first, so it
works with an outdated runtime too. In your own scripts, import ``pyRadPlan.gui``
before other pyRadPlan modules (such as ``pyRadPlan.io``) to get the same effect. If Qt still
fails to load, pyRadPlan now shows the original error and points to the Visual C++
Redistributable instead of asking to install the ``gui`` extra.

About the project
-----------------

pyRadPlan is an open-source, multi-modality radiotherapy treatment planning toolkit,
born from the established MATLAB-based toolkit `matRad <http://www.matRad.org>`_.
Development is lead by the `Radiotherapy Optimization group <https://www.dkfz.de/radopt>`_
at the `German Cancer Research Center (DKFZ) <https://www.dkfz.de>`_.
