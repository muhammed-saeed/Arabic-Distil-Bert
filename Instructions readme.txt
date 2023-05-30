1-training DistilBert to understand Arabic Language using transformers, the code is run in google cloud services
2-TFRC TPUS v3.8 note we are running codes based on Pytorch
3-Modern Standard Arabic Data used for training is Oscar Arabic data the 81 Gigabyte version, we have this very version in our google cloud bucket since we used it
to train our Sudanese Bert Model
4-The sudanese Data is collected in previous research titled:SudaBert
5-

///////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/////////////////\\\\\\\\\\\\\\\
downloading the arabic data set note that the data is hosted in HumanID Oscar and needs to send them email with univesrity.edu to get access to teh data

to downlaod and authenticate the data 
/home/sudanese_distilbert/files_ar/oscar-prive.huma-num.fr/2109/packaged/ar

wget --user=USERNAME --ask-password https://oscar-prive.huma-num.fr/2109/packaged/ar/ar_part_1.txt.gz

to donwload the entire directory
wget -r --user=USERNAME --ask-password https://oscar-prive.huma-num.fr/2109/packaged/ar/






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
1- 

how to setup GCP
In google cloud go to compute engine --> then choose VM instance then choose create instance --> 
i-make sure the name of the instance is the same as the name of the TPU you will create later on in our case the name is "distilbert-pretraining"
ii- for our case the tpu's were free at europe-west4(Netherlands) so i choose this as the region
iii- choosing custom machine with 8 CPU's and 16 Gigabyte ram
iv-choosing desk HDD with 1.7tera maximum allowed with TPU in our case
v-Operating system --> choose deep learning on linux and the version choose pytorch/Xla
usually its at the bottom of the list

##################################################

Note now there are more updated versions of pytorch_xla, in the past we have to use debian_9 pytorch_xla; 
however now one can use pytorch_xla_1.8 fastai packages




/////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////////
2-
Install pip and pip3

install pip:

sudo apt update
sudo apt install python-pip
pip --version


install pip3:

sudo apt update
sudo apt install python3-pip
pip3 --version



\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

3-

how to setup pytorch TPU
Step1:

Create and Intiate the TPU using gcloud command:
 gcloud compute tpus create distilbert-pretraining \
--zone=europe-west4-a \
--network=default \
--version=pytorch-1.6 \
--accelerator-type=v3-8

          -----------------------------------------
Step2:
then check its IP address using:
gcloud compute tpus list --zone=europe-west4-a
            --------------------------------------
Step 3:
activate conda enviroment named torch-xla-nightly:
conda activate torch-xla-nightly
              ------------------------
            
Step 4:            
add TPU configuration to the VM path (replace $IP_ADDRESS$ with the ip address of your TPU from step 2):
export XRT_TPU_CONFIG="tpu_worker;0;$IP_ADDRESS$:8470"
                  -------------------------------
Step 5:
to check if the TPU is created and working with pytorch then open terminal in google cloud and type python3, after it type the below python code

            ==============
import torch
import torch_xla
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
t1 = torch.ones(3, 3, device = dev)
print(t1) 
              ============

Note: if the output is 3x3 array filled with ones then your TPU is correctly created and working with pytorch



\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\
4- 
Download the latest version of datasets,likewise, of transformers using "git"  and install it by the following commands

git clone https://github.com/huggingface/datasets.git
cd datasets
pip3 install -e .
cd ~
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////////////\\\\\\\\\\\\\\\\////////////////////////\\\\\\\\\\\\\
5-run the transformers/examples/language-modeling/run_mlm.py  by typing the following command on terminal

python3 transformers/examples/xla_spawn.py --num_cores=8 \
   transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path init_distilbert\
    --train_file 30gig_files/ar_file1.txt \
    --do_train \
    --do_eval  \
    --output_dir output \
    --validation_split_percentage=2 \
    --per_device_eval_batch_size=2048 \
    --num_train_epochs=1 \
    --line_by_line
    
  Note we are dividing the 81 gigabyte oscar data into three chunks
    
  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\///////////////////\\\\\\\\\\\\\\\\\\\\\\\\\///////////
  6-
