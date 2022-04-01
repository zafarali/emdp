.. emdp documentation master file, created by
   sphinx-quickstart on Thu Mar 31 15:59:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome
================================

Welcome to emdp's documentation!

Easy MDPs implemented in a gym like interface with access to transition dynamics.

Background
-------------------------------

MDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Markov Decision Process (MDP) is defined as a five tuple
:math:`(\mathcal{S} ,\mathcal{A} ,r ,P ,\gamma)`,
where :math:`\mathcal{S}`` is a set of states, :math:`\mathcal{A}` is a set of actions,
:math:`r:\mathcal{S}\times\mathcal{A} \mapsto \mathbb{R}` is a reward function,
:math:`P:\mathcal{S}\times\mathcal{A}\mapsto \Pr(\mathcal{S})` is a state transition function,
and :math:`\gamma \in (0,1]` is a discount factor.


Contents
=================

.. toctree::
   :maxdepth: 2

   tutorial/index
   topic/index
   api/emdp

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
