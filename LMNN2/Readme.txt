*** Changelog
Update 09/19/2012
- Added 32-bit Windows binaries (Thanks to Ya Shi)
Update 09/18/2012
- Added parameter 'outdim' to easily specify the output dimensionality
- Small fixes in mtree code, which broke compilation on some windows machines. 
- Speedup in findimps3Dm by substituting some repmats with bsxfun (somehow they have been overlooked)

Update 09/13/2012
- Small fix to setpaths.m script
- Rearranged files to ensure that the mexed files are in the path.
- updated demo

Update 09/06/2012
- Small fix to instlall.m script


Update 08/23/2012

This package contains the implementation of Large Margin Nearest Neighbors (LMNN). 

Changes from version 2.0 to 2.1:
 - Removed mex files which are no longer faster than the Matlab equivalent (Matlab became a lot faster over the years)
 - Updated mtrees to compile on windows computers and no longer use depreciated libraries
 - Removed all BLAS / LAPACK dependencies 
 - Renamed knnclassify.m to knncl.m (as former clashed with the implementation from the statistics toolbox)
(Many thanks to Jake Gardner who helped a lot with tyding up of the code.)



*** Usage
Please run (inside the matlab console)
install
demo

to see a working demo. 




If you use this code in scientific work, please cite:

@article{weinberger2009distance,
  title={Distance metric learning for large margin nearest neighbor classification},
  author={Weinberger, K.Q. and Saul, L.K.},
  journal={The Journal of Machine Learning Research},
  volume={10},
  pages={207--244},
  year={2009},
  publisher={JMLR.org}
}
