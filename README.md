
## MirrorFlow: Exploiting Symmetries in Joint Optical Flow and Occlusion Estimation


### External Dependencies

* OpenCV (tested on 2.4.13, 3.0.0, and 3.2.0)
* png++ (tested on 0.2.7)
* libpng (tested on 1.6.26)
  
  
### Input arguments

* prefix input_image1 input_image2 result_directory parameter_directory parameter_file initial_forward_flow initial_backward_flow
* ex.) 000064 ../data/000064_10.png ../data/000064_11.png ../data/res/ ../params/ params_gco.yml ../data/init/000064_10.png ../data/init/000064_11.png


### Parameter files 

* params_gco.yml
    * A set of main parameters
    * Details are described in the main paper.	
	
* params_mirrorFlow.yml
    * Controlling verbose options.   
	* verbose_main: a verbose flag for the main algorithm
	* verbose_seg: a verbose flag for the superpixel algorithm
	
* params_tgseg.yml
    * A set of parameters for the initial superpixelization (related to the paper below)
    * CVPR 2015: Real-Time Coarse-to-fine Topologically Preserving Segmentation
    * http://www.cs.toronto.edu/~yaojian/cvpr15.pdf

	
### Included Dependencies

The source codes contains the following works which they include their own licenses:

* Spixel:	
	* Real-time coarse-to-fine topologically preserving segmentation
	* Project link: https://bitbucket.org/mboben/spixel
    * Contact information: marko.boben@fri.uni-lj.si
	* Distributed under GNU GPL version 3.0
	* Citation: 
		* J. Yao, M. Boben, S. Fidler, and R. Urtasun. Real-time coarse-to-fine topologically preserving segmentation. In CVPR, pages 2947-2955, 2015.

* GCoptimization:	
	* GCO-v3.0: software for energy minimization with graph cuts
	* Project link: http://vision.csd.uwo.ca/code/
	* Contact information: olga@csd.uwo.ca
	* Citations:
		* Y. Boykov, O. Veksler, R.Zabih, "Efficient Approximate Energy Minimization via Graph Cuts," IEEE TPAMI, 20(12):1222-1239, Nov 2001.
		* V. Kolmogorov, R.Zabih, "What Energy Functions can be Minimized via Graph Cuts?," IEEE TPAMI, 26(2):147-159, Feb 2004. 
		* Y. Boykov, V. Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision," IEEE TPAMI, 26(9):1124-1137, Sep 2004.
		* R. Zabih, Y. Boykov, O. Veksler, "System and method for fast approximate energy minimization via graph cuts," United Stated Patent 6,744,923, June 1, 2004

* QPBO:
	* Implements algorithms for minimizing functions of binary variables with unary and pairwise terms based on roof duality described in the following papers
	* Project link: http://pub.ist.ac.at/~vnk/
	* Contact information: vnk@ist.ac.at
	* Citations:
		* P. L. Hammer, P. Hansen, and B. Simeone, "Roof duality, complementation and persistency in quadratic 0-1 optimization," Mathematical Programming, 28:121-155, 1984.
		* E. Boros, P. L. Hammer, and X. Sun, "Network flows and minimization of quadratic pseudo-Boolean functions," Technical Report RRR 17-1991, RUTCOR Research Report, May 1991.
		* E. Boros, P. L. Hammer, and G. Tavares, "Preprocessing of Unconstrained Quadratic Binary Optimization," Technical Report RRR 10-2006, RUTCOR Research Report, April 2006.
		* C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer, "Optimizing binary MRFs via extended roof duality," CVPR 2007.

* KITTI development kit:
	* KITTI development kit for handling flow data format
	* Project link: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow 
	* Contact information: andreas.geiger@tuebingen.mpg.de
	* Citations:
		* M. Menze and A. Geiger, "Object scene flow for autonomous vehicles," In CVPR, pages 3061-3070, 2015.
		
		
### License

See [LICENSE.md](LICENSE.md) for details

 
### Citation

Please cite the paper below if you find our paper/source code are useful.  

    @inproceedings{Hur:2017:MFE,  
      Author = {Junhwa Hur and Stefan Roth},  
      Booktitle = {ICCV},  
      Title = {{MirrorFlow}: {E}xploiting Symmetries in Joint Optical Flow and Occlusion Estimation},  
      Year = {2017}  
    }
	
