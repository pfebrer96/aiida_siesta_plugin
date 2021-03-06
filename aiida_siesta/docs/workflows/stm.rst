SIESTA STM workflow
++++++++++++++++++++++

Description
-----------

The **SiestaSTMWorkchain** workflow consists in 3 steps:

* Performing of a siesta calculation on an input structure (including relaxation if needed) 
  through the **SiestaBaseWorkChain**.
* Performing of a further siesta calculation aimed to produce a .LDOS file.
* A call to the `plstm` code to post process the .LDOS file and
  create simulated STM images. The call is made via the
  STMCalculation plugin, which is also included in the `aiida_siesta` distribution.

The .LDOS file contains informations on the local density
of states (LDOS) in an energy window. The LDOS can be seen as a
"partial charge density" to which only those wavefunctions with
eigenvalues in a given energy interval contribute. In the
Tersoff-Hamann approximation, the LDOS can be used as a proxy for the
simulation of STM experiments. The 3D LDOS file is then processed by the
specialized program `plstm` to produce a 2D section in "constant-height" or 
"constant-current" mode, optionally projected on spin components
(see the header/manual for plstm, and note that non-collinear and spin-orbit 
modes are supported). 
The "constant-height" mode corresponds to the creation of 
a plot of the LDOS in a 2D section at a given height in the unit cell 
(simulating the height of a STM tip). The "constant-current" mode
simulates the topography map by recording the z
coordinates with a given value of the LDOS.

The inputs to the STM workchain include all the inputs of the **SiestaBaseWorkChain**
to give full flexibility on the choice of the siesta calculation
parameters. The energy window for the LDOS is specified respect to the Fermi energy.
In fact, a range of
energies around the Fermi Level (or regions near to the HOMO and/or
LUMO) are the meaninful energies for the STM images production. 
The tip height ("constant-height" mode) or the LDOS iso-value ("constant-current" mode)
must be specified by the user in input.
The workchain returns an AiiDA ArrayData object whose
contents can be displayed by standard tools within AiiDA and the wider
Python ecosystem.


Supported Siesta versions
-------------------------

At least 4.0.1 of the 4.0 series, and 4.1-b3 of the 4.1 series, which
can be found in the development platform
(https://gitlab.com/siesta-project/siesta/).

Inputs
------

* All the inputs of the **SiestaBaseWorkChain**, as explained
  :ref:`here <siesta-base-wc-inputs>`.

.. |br| raw:: html

    <br />

* **stm_code**, class :py:class:`Code  <aiida.orm.Code>`, *Mandatory*

  A code associated to the STM (plstm) plugin (siesta.stm). See plugin documantation for more details.

.. |br| raw:: html

    <br />

* **stm_mode**, class :py:class:`Str <aiida.orm.Str>`, *Mandatory*

  Allowed values are `constant-height` or `constant-current`, corresponding to the two
  operation modes of the STM that are supported by the plstm code.

.. |br| raw:: html

    <br />


* **stm_value**, class :py:class:`Float <aiida.orm.Float>`, *Mandatory*

  The value of height or current at which the user wants to simulate the
  STM. This value represents the tip height in "constant-height" mode
  or the LDOS iso-value in "constant-current" mode.
  The height must be expressed in Ang, the current in e/bohr**3.

.. |br| raw:: html

    <br />


* **emin**, class :py:class:`Float  <aiida.orm.Float>`, *Mandatory*

  The lower limit of the energy window for which the LDOS is to be
  computed (in eV and respect to the Fermi level).

.. |br| raw:: html

    <br />

* **emax**, class :py:class:`Float <aiida.orm.Float>`, *Mandatory*

  The upper limit of the energy window for which the LDOS is to be
  computed (in eV and respect to the Fermi level).

.. |br| raw:: html

    <br />

* **stm_spin**, class :py:class:`Str <aiida.orm.Str>`, *Mandatory*

  Allowed values are `none`, `collinear` or `non-collinear`.
  Please note that this keyword only influences the STM post process!
  It does not change the parameters of the siesta calculation, that must
  be specified in the `parameters` input port.
  In fact, this keyword will be automatically reset if a `stm_spin`
  option incompatible with the parent siesta spin option is chosen.
  A warning will be issued in case this happens.
  This keyword also influences the structure of the output port
  `stm_array`. If fact, if the `non-collinear` value is chosen, the
  workflow automatically performs the STM analysis in the three
  spin components and for the total charge option, resulting in a
  richer `stm_array` (see description in the Outputs section).

.. |br| raw:: html

    <br />

* **stm_options**, class :py:class:`Dict <aiida.orm.Dict>`, *Optional*
  
  This dictionary can be used to specify the computational resources to
  be used for the STM calculation (the `plstm` code). It is optional
  because, if not specified, the same resources of the siesta calculations
  are used, except that the parallel options are stripped off.
  In other words, by default, the `plstm` code runs on a single processor. 

..
        * **protocol**, class :py:class:`Str <aiida.orm.Str>`

        Either "standard" or "fast" at this point.
        Each has its own set of associated parameters.

        - standard::

             {
                'kpoints_mesh_offset': [0., 0., 0.],
                'kpoints_mesh_density': 0.2,
                'dm_convergence_threshold': 1.0e-4,
                'forces_convergence_threshold': "0.02 eV/Ang",
                'min_meshcutoff': 100, # In Rydberg (!)
                'electronic_temperature': "25.0 meV",
                'md-type-of-run': "cg",
                'md-num-cg-steps': 10,
                'pseudo_familyname': 'lda-ag',
                'atomic_heuristics': {
                    'H': { 'cutoff': 100 },
                    'Si': { 'cutoff': 100 }
                },
                'basis': {
                    'pao-energy-shift': '100 meV',
                    'pao-basis-size': 'DZP'
                }
	      }

        - fast::
    
             {
                'kpoints_mesh_offset': [0., 0., 0.],
                'kpoints_mesh_density': 0.25,
                'dm_convergence_threshold': 1.0e-3,
                'forces_convergence_threshold': "0.2 eV/Ang",
                'min_meshcutoff': 80, # In Rydberg (!)
                'electronic_temperature': "25.0 meV",
                'md-type-of-run': "cg",
                'md-num-cg-steps': 8,
                'pseudo_familyname': 'lda-ag',
                'atomic_heuristics': {
                    'H': { 'cutoff': 50 },
                    'Si': { 'cutoff': 50 }
                },
                'basis': {
                    'pao-energy-shift': '100 meV',
                    'pao-basis-size': 'SZP'
                }
	      }

        The *atomic_heuristics* dictionary is intended to encode the
        peculiarities of particular elements. It is work in progress.

        The *basis* section applies globally for now.


Outputs
-------

* **stm_array** :py:class:`ArrayData <aiida.orm.ArrayData>` 

  In case the `stm_spin` is `none` or `collinear` this output port
  is a collection of three 2D arrays (`grid_X`, `grid_Y`, `STM`) holding the section or
  topography information. Exactly like the output of the STM plugin.
  In case the `stm_spin` is `non-collinear`, this output port
  is a collection of six 2D arrays (`grid_X`, `grid_Y`, `STM_q`, `STM_sx`, `STM_sy`, `STM_sz`)
  holding the section or topography information for the total charge STM analysis and 
  the three spin components.
  Both cases follow the `meshgrid` convention in
  Numpy. A contour plot can be generated with the `get_stm_image.py`
  script in the repository of examples. The `get_stm_image.py` script
  automatically detects how many arrays are in `stm_spin`, therefore it is 
  completely general.

.. |br| raw:: html

    <br />

* **output_structure** :py:class:`StructureData <aiida.orm.StructureData>`

  Present only if the siesta calculation is moving the ions.  Cell and ionic
  positions refer to the last configuration, on which the STM analysis is performed.

  



