NOTES:

----------------------------------
* SETUP:
----------------------------------
+ run setup for setpath and mex C files


----------------------------------
* DEMO:
----------------------------------
+ demoAIT: run Generalized Aitchison Embeddings for MIT Scene dataset
+ demoLMNN: run LMNN for MIT Scene dataset
(if flagHELLINGER = 1, we have LMNN with Hellinger mapping)

Using data in "Data" folder, results are also saved in this one.
+ 100 images for each class in training and 100 other images for each class for test


----------------------------------
* FUNCTIONS in LIB:
----------------------------------
+ AIT_PSGD_NES: projected sugradient descent with Nesterov's acceleration for learning generalized Aitchison embeddings
+ Third party toolbox (LMNN2): LMNN solver from Weinberger (with some modifications)
(Link: http://www.cse.wustl.edu/~kilian/code/page21/page21.html)
+ InitLOption: contains some options to initialize L (linear transform matrix)
(For pseudo-count vector, without prior information, we use uniform vectors.)
+ Utility: contains some helping function for AIT_PSGD_NES


----------------------------------
* DATA:
----------------------------------
% Data folder
% MIT Scene dataset (8 classes of scene images)
% Using bag-of-feature to construct histograms for each image
%   1. Using SIFT feature on each pixel.
%   2. Building dictionary by Kmeans (set K=100 --> we have 100 visual words in dictionary)
%   3. Using k-dTree to map SIFT to visual word.
%   4. Computing frequency of each visual word in an image to construct histogram representation


----------------------------------
RELEVANT PAPERS:
----------------------------------
% [1] Tam Le, Marco Cuturi, Generalized Aitchison Embeddings for Histograms,
% Asian Conference on Machine Learning (ACML), 2013.
% [2] Tam Le, Marco Cuturi, Generalized Aitchison Embeddings for
% Histograms, Machine Learning Journal (MLJ), 2014.


----------------------------------
* CONTACT
----------------------------------
% Version 0.1 (May 9th, 2014)
@ Tam Le - Kyoto University
(tam.le@iip.ist.i.kyoto-u.ac.jp)


Please contact me if you observe any bugs in the execution of the algorithms.
Many thanks !!!



