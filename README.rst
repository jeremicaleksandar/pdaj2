About
=====

A distributed system for computing and exporting beam integrals of all 6
supported beam types, as defined in the `beam_integrals project`_ and
[Milasinovic1997]_.

This work is a part of the investigation within the research projects:
[ON174027]_ and [TR36017]_, supported by the Ministry for Science and
Technology, Republic of Serbia. This support is gratefully acknowledged.

References
----------

.. [Milasinovic1997]
   Milašinović, D.D. "The Finite Strip Method in Computational Mechanics".
   Faculties of Civil Engineering: University of Novi Sad, Technical University
   of Budapest and University of Belgrade: Subotica, Budapest, Belgrade. (1997)
.. [ON174027]
   "Computational Mechanics in Structural Engineering"
.. [TR36017]
   "Utilization of by-products and recycled waste materials in concrete
   composites in the scope of sustainable construction development in Serbia:
   investigation and environmental assessment of possible applications"

.. _`beam_integrals project`: https://bitbucket.org/petar/beam_integrals


Installation
============

export_beam_integrals doesn't require system wide installation, just clone this
repository to get started.


Quick start (distributed system in a box)
=========================================

Make any changes you'd like to ``Vagrantfile`` and ``docker-compose.yml``,
and please note that environment variables are used to configure the distributed
system. See ``Dockerfile.server`` and ``Dockerfile.worker`` for more details.

Run the following command will automatically provision a new virtual machine;
which will then build, configure and start the entire distributed system::

    $ vagrant up


Production use
==============

export_beam_integrals has been thoroughly tested and is production ready, either
by using standard Docker tools or specialized automated clustering systems. See
``Dockerfile.server`` and ``Dockerfile.worker`` for more details.


Contribute
==========

If you find any bugs, or wish to propose new features `please let us know`_. 

If you'd like to contribute, simply fork `the repository`_, commit your changes
and send a pull request. Make sure you add yourself to `AUTHORS`_.

.. _`please let us know`: https://bitbucket.org/petar/export_beam_integrals/issues/new
.. _`the repository`: http://bitbucket.org/petar/export_beam_integrals
.. _`AUTHORS`: https://bitbucket.org/petar/export_beam_integrals/src/default/AUTHORS
