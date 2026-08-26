bio_models
==========

.. currentmodule:: pyRadPlan.bio_models

Models
------

.. autosummary::
   :toctree: generated/bio_models/

   BiologicalModel
   EmptyModel
   ConstantRBEModel
   LQModel
   LETBasedLQModel
   RBEMinMax
   Wedenberg
   MCNamara
   Carabe
   HeliumMairani
   LinearScaling
   KernelBasedLQModel
   TabulatedRBEModel
   TabulatedAlphaBetaModel


Evaluators and tissue lookups
-----------------------------

.. autosummary::
   :toctree: generated/bio_models/

   BioModelEvaluator
   ParametricEvaluator
   KernelBasedEvaluator
   TabulatedSpectrumEvaluator
   TissueParameterLookup
   ExactClassLookup
   make_tissue_lookup


Creating and registering models
-------------------------------

.. autosummary::
   :toctree: generated/bio_models/

   create_bio_model
   get_bio_model
   available_bio_models
   get_available_models
   register_model
   bio_model_spec_from_matrad
   alpha_beta_influence_from_let
