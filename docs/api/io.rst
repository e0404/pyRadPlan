io
===

.. currentmodule:: pyRadPlan.io

Loading and saving
------------------

.. autosummary::
   :toctree: generated/io/

   load_patient
   load_data
   save_data
   ExportBundle
   load_tg119
   list_phantoms
   available_phantoms
   phantom_path
   load_binary_patient
   list_image_files
   validate_matrad_patient


Format handlers
---------------

.. autosummary::
   :toctree: generated/io/

   MatlabHandler
   DicomHandler
   NpzHandler
   PickleHandler
   Hdf5Handler
   NiftiHandler
   NrrdHandler
   MetaImageHandler


Extending
---------

.. autosummary::
   :toctree: generated/io/

   BaseImporter
   BaseExporter
   register_importer
   register_exporter
   get_importer
   get_exporter
   get_available_formats
