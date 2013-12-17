

----------------------------------------------------
Johns Hopkins University Ionosphere database

Source Information:
   -- Donor: Vince Sigillito (vgs@aplcen.apl.jhu.edu)
   -- Date: 1989
   -- Source: Space Physics Group
              Applied Physics Laboratory
              Johns Hopkins University
              Johns Hopkins Road
              Laurel, MD 20723 

The relevant paper is:
Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266. 

----------------------------------------------------

For more detail attribute description and source information, see "ionosphere.names.txt" 



=========================================================================
The USPS digits data were gathered at the Center of Excellence in Document Analysis and Recognition (CEDAR) at SUNY Buffalo, as part of a project sponsored by the US Postal Service. The dataset is described in A Database for Handwritten Text Recognition Research, J. J. Hull, IEEE PAMI 16(5) 550-554, 1994.
=========================================================================

There are two datasets available:

- usps_all.mat       is the original sups dataset

- usps_resampled.mat is collected in a different way of original dataset.
                     See http://www.gaussianprocess.org/gpml/data/

=========================================================================

If you want to test on binary classification tasks, there are also two scripts for conveniently load databases with only selecting two classes:

- [Matlab]loadBinaryUSPS.m     
  Copyright (C) 2005 and 2006, Carl Edward Rasmussen

- [Python]loadBinaryUSPS.py
  Copyright (C) 2013, Shan Huang

With slight changes, you can use these scripts to load any subsets of dataset you want to evaluate on.

