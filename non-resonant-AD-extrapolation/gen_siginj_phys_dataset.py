import argparse
import numpy as np
from math import sin, cos, pi
from helpers.plotting import *
from helpers.process_data import *
from semivisible_jet.utils import *
from sklearn import preprocessing
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigsample",help="Input signal .txt file",
    default="../working_dir/samples/sig_samples/rinv13_3TeV_300GeV.txt"
)
parser.add_argument("-b1","--bkg-dir",help="Input bkground folder",
    default="../working_dir/samples/qcd_data_samples/"
)
#parser.add_argument("-b2","--ideal-bkg-dir",help="Input ideal bkground folder",
#    default="../working_dir/qcd_data_samples/"
#)
parser.add_argument("-mc", "--mc-dir",help="Input MC bkground folder",
    default="../working_dir/samples/qcd_data_samples/"
)
parser.add_argument("-size",type=int,help="Number of bkg text files",default=20)
parser.add_argument("-o","--outdir",help="output directory", default = "../working_dir/")
parser.add_argument("-make_static",action='store_true',help="Whether to make the static samples")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-morph_mc",action='store_true',help="Whether to tamper with the MC to make it look more different from the data")


args = parser.parse_args()

def main():

    # Create the output directory
    data_dir = f"{args.outdir}/data/"
    os.makedirs(data_dir, exist_ok=True)

    # define sample size as the number of files
    sample_size = args.size

    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]

    # First load in signal
    variables, sig = load_samples(args.sigsample)
    sig = get_quality_events(sig)
    sig_events = sort_event_arr(var_names, variables, sig)

    # Datasets that don't change: ideal bkg, MC
    if args.make_static:

        print(f"Loading {sample_size} samples of mc...")

        ideal_bkg_events_list = []
        mc_events_list = []

        for i in range(sample_size):

           # ideal_bkg_path = f"{args.ideal_bkg_dir}/qcd_{i}.txt"
            mc_path = f"{args.mc_dir}/qcd_{i}.txt"

           #if os.path.isfile(ideal_bkg_path) and os.path.isfile(mc_path):
            if os.path.isfile(mc_path):

                # Load input events ordered by varibles
               # _, ideal_bkg_i = load_samples(ideal_bkg_path)
                _, mc_i = load_samples(mc_path)

               # ideal_bkg_i = get_quality_events(ideal_bkg_i)
                mc_i = get_quality_events(mc_i)

                #ideal_bkg_events_list.append(sort_event_arr(var_names, variables, ideal_bkg_i))
                mc_events_list.append(sort_event_arr(var_names, variables, mc_i))
                
            else:
                #check_file_log(ideal_bkg_path, mc_path)
                check_file_log(mc_path)

        if (len(mc_events_list)==0):
            sys.exit("No files loaded. Exit...")

        print("Done loading!")

        # concatenate all background events
        #ideal_bkg_events = np.concatenate(ideal_bkg_events_list)
        mc_events = np.concatenate(mc_events_list)
        
        if args.morph_mc:
            print("Morphing the mc events a bit...")
            mc_events = morph_mc(mc_events)
        
        # preprocess data -- fit to MC
        scaler = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit(mc_events)
        
        with open(f"{data_dir}/mc_scaler.pkl","wb") as f:
            print("Saving out trained minmax scaler.")
            pickle.dump(scaler, f)
            
        # SR masks
        mc_mask_SR = phys_SR_mask(mc_events)
        mc_mask_CR = ~mc_mask_SR

       # ideal_bkg_mask_SR = phys_SR_mask(ideal_bkg_events)
       # ideal_bkg_mask_CR = ~ideal_bkg_mask_SR
        
        # save out the events that don't change with signal injection
        np.savez(f"{data_dir}/mc_events.npz", mc_events_cr=scaler.transform(mc_events[mc_mask_CR]), mc_events_sr=scaler.transform(mc_events[mc_mask_SR]))
       # np.savez(f"{data_dir}/ideal_bkg_events.npz", ideal_bkg_events_cr=scaler.transform(ideal_bkg_events[ideal_bkg_mask_CR]), ideal_bkg_events_sr=scaler.transform(ideal_bkg_events[ideal_bkg_mask_SR]))
        
    else: 
        with open(f"{data_dir}/mc_scaler.pkl","rb") as f:
            print("Loading in trained minmax scaler.")
            scaler = pickle.load(f)
    
    print(f"Loading {sample_size} samples of bkg (and some signal)...")

    bkg_events_list = []
    for i in range(sample_size):
        bkg_path = f"{args.bkg_dir}/qcd_{i}.txt"
        if os.path.isfile(bkg_path):  
            # Load input events ordered by varibles
            _, bkg_i = load_samples(bkg_path)
            bkg_i = get_quality_events(bkg_i)
            bkg_events_list.append(sort_event_arr(var_names, variables, bkg_i))
        else:
            check_file_log(bkg_pathh)
    if len(bkg_events_list)==0:
        sys.exit("No files loaded. Exit...")
        
    print("Done loading!")

    # concatenate all background events
    bkg_events = np.concatenate(bkg_events_list)

    # SR masks
    bkg_mask_SR = phys_SR_mask(bkg_events)
    bkg_mask_CR = ~bkg_mask_SR
    
    # Create folder for the particular signal injection
    seeded_data_dir = f"{data_dir}/seed{args.gen_seed}/"
    os.makedirs(seeded_data_dir, exist_ok=True)
    np.random.seed(int(args.gen_seed))
 
    # initialize lists
    sig_percent_list = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024] # 4TeV
    #sig_percent_list = [0, 0.006, 0.012, 0.018, 0.024, 0.03, 0.036] # 2TeV
    #sig_percent_list = [0, 0.004, 0.009, 0.013, 0.018, 0.022, 0.027] # 3TeV
    #sig_percent_list = [0, 0.0031, 0.0065, 0.0093, 0.012, 0.015, 0.018]
    # Create signal injection dataset
    n_bkg_SR = bkg_events[bkg_mask_SR].shape[0]
    
    for s in sig_percent_list:
        
        # Subsample signal set
        n_sig = int(s * n_bkg_SR)
        selected_sig_indices = np.random.choice(sig_events.shape[0], size=n_sig, replace=False)
        selected_sig = sig_events[selected_sig_indices, :] 

        # Create data arrays
        data_events = np.concatenate([selected_sig, bkg_events])

        # SR masks
        data_mask_SR = phys_SR_mask(data_events)
        data_mask_CR = ~data_mask_SR
        selected_sig_mask_SR = phys_SR_mask(selected_sig)

        # SR events
        n_sig_SR = selected_sig[selected_sig_mask_SR].shape[0]
        s_SR = round(n_sig_SR/n_bkg_SR, 5)
        signif = round(n_sig_SR/np.sqrt(n_bkg_SR), 5)

        # Print dataset information
        print(f"S/B={s_SR} in SR, S/sqrt(B) = {signif}, N bkg SR: {n_bkg_SR:.1e}, N sig SR: {n_sig_SR}")
        
        # Plot varibles in the SR
        
        sig_list = selected_sig[selected_sig_mask_SR].T
        bkg_list = bkg_events[bkg_mask_SR].T
        data_list = data_events[data_mask_SR].T
        plot_dir = f"{seeded_data_dir}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        # Signal vs background
        plot_kwargs = {"name":f"sig_vs_bkg_SR_{s}", "title":"Signal vs background in SR", "outdir":plot_dir}
        plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)
        # data vs background SR
        plot_kwargs = {"labels":["data", "bkg"], "name":f"data_vs_bkg_SR_{s}", "title":"Data vs background in SR", "outdir":plot_dir}
        plot_all_variables(data_list, bkg_list, var_names, **plot_kwargs)
        
        # Save dataset
        np.savez(f"{seeded_data_dir}/data_{s}.npz", data_events_cr=scaler.transform(data_events[data_mask_CR]), data_events_sr=scaler.transform(data_events[data_mask_SR]), sig_percent=s_SR)
        
    
    print(f"Finished generating dataset. (Gen seed: {args.gen_seed})")

if __name__ == "__main__":
    main()
