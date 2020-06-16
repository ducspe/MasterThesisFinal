# This code accompanies the master thesis "Decentralized Information Gathering with Continuous and very large Discrete Observation Spaces"
# It is used for:
## 1) Modeling the Variational Autoencoder that compresses the images for the encoded COGNet-DICE algorithm
## 2) Generating the dataset for training the Variational Autoencoder
## 3) Training the Variational Autoencoder with the generated training set
# It is advisable to have a system enabled by a GPU

# To generate train and test datasets, run "training_data_generator.py"
# To train the VAE, make sure "perform_training(...)" is uncommented inside "vision_patch_vae.py" and then run "vision_patch_vae.py"
# Current VAE weights and training dataset are available in "datafolder" 