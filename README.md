# GraphDataMining

The code consists of the simulation framework and Python script for the GCN-LSTM model.

From the simulation, I have attached the .cc file where the Python script is called. The GCN-LSTM model's Python script and HO decision-making are done in the function "void LtePhyUe::handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo)" in "LtePhyUe.cc" file.

Required packages: pip install pandas torch torchvision torchaudio torch-geometric scikit-learn

You can run the Python script as alone. For myself, I called the Python script from the "void LtePhyUe::handoverHandler(LteAirFrame* frame, UserControlInfo* lteInfo)" function and data captured there in a csv file.
