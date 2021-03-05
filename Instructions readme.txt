1-training DistilBert to understand Arabic Language using transformers, the code is run in google cloud services
2-TFRC TPUS v3.8 note we are running codes based on Pytorch
3-Modern Standard Arabic Data used for training is Oscar Arabic data the 81 Gigabyte version, we have this very version in our google cloud bucket since we used it
to train our Sudanese Bert Model
4-The sudanese Data is collected in previous research titled:SudaBert

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
how to setup GCP
In google cloud go to compute engine --> then choose create instance --> the 



/////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////////
Install pip and pip3

nstall pip3
sudo apt update
sudo apt install python3-pip
pip3 --version



\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
how to setup pytorch TPU
Step1L

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
