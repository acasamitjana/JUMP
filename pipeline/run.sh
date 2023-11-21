#python initialization/bold2mri.py
#python initialization/synthseg.py --threads 16
#python preprocess/anat_preproc.py

#python registration/initialize_graph.py
#python registration/compute_graph.py

python preprocess/bold_preproc.py

python registration/register_template.py

#python preprocess/bold_preproc.py --group_melodic --force
