import uproot
import vector 
import awkward as ak

def create_parquet(filename,path):
    
    f = uproot.open(path+filename)
    
    top=vector.zip({
    "pt":f["truth"]["truth_parent_pt"].array()[:,0],
    "eta":f["truth"]["truth_parent_eta"].array()[:,0],
    "phi":f["truth"]["truth_parent_phi"].array()[:,0],
    "mass":f["truth"]["truth_parent_m"].array()[:,0],
    })

    antitop=vector.zip({
        "pt":f["truth"]["truth_parent_pt"].array()[:,1],
        "eta":f["truth"]["truth_parent_eta"].array()[:,1],
        "phi":f["truth"]["truth_parent_phi"].array()[:,1],
        "mass":f["truth"]["truth_parent_m"].array()[:,1],
    })

    reco_arr = f["reco"].arrays(f["reco"].keys())
    
    # rename "njet" "global_njet" and remove "njet"
    for field in ["njet", "nelectron", "nmuon", "nbtagged", "met_met","met_phi"]:
        reco_arr[f"global_{field}"] = reco_arr[field]
        del reco_arr[field]
        
    # for field in ["jet_pt", "jet_eta", "jet_phi", "jet_mass"]:
    #     reco_arr[f"jet.{field}"] = reco_arr[field]
    #     del reco_arr[field]
    
    gen_el = vector.zip({"pt":f["truth"]["gen_el_pt"].array(),
                     "eta":f["truth"]["gen_el_eta"].array(),
                     "phi":f["truth"]["gen_el_phi"].array(),
                     "mass":f["truth"]["gen_el_m"].array()})

    gen_mu = vector.zip({"pt":f["truth"]["gen_mu_pt"].array(),
                        "eta":f["truth"]["gen_mu_eta"].array(),
                        "phi":f["truth"]["gen_mu_phi"].array(),
                        "mass":f["truth"]["gen_mu_m"].array()})

    leps = ak.concatenate([gen_el,gen_mu],axis=1)
    charges = ak.concatenate([f["truth"]["gen_el_charge"].array(),f["truth"]["gen_mu_charge"].array()],axis=1)
    sorted_leps = leps[ak.argsort(charges,ascending=False)]
    gen_lP = sorted_leps[:,0]
    gen_lM = sorted_leps[:,1]
    
    from hepsetreg.physics.reco_spins import boost, helicity_basis_observables
    
    hel_obs = helicity_basis_observables(*boost(top,antitop,gen_lP,gen_lM))
        
    target = ak.Array({"ttbar_mass":(top+antitop).m,
              "ttbar_pt":(top+antitop).pt,
              "ttbar_DR":top.deltaR(antitop),
              "top_px":top.px,
              "top_py":top.py,
                "top_pz":top.pz,
                "top_E":top.E,
                "antitop_px":antitop.px,
                "antitop_py":antitop.py,
                "antitop_pz":antitop.pz,
                "antitop_E":antitop.E,
              })
    
    # Add the helicity basis observables to the target
    target = ak.zip({**{f: target[f] for f in target.fields}, 
                     **{f: hel_obs[f] for f in hel_obs.fields}}, 
                    depth_limit=1)

    combined = ak.zip({"inputs": reco_arr, "targets": target}, depth_limit=1)
    
    ak.to_parquet(combined, f"{filename.replace(".root","")}_051026.parquet")
    
if __name__ == "__main__":
    path = "/mnt/iusers01/jw01/c73320es/data/HanData/"
    print("Running")
    for filename in ["ttbar-2L_test.root", "ttbar-2L_train.root", "ttbar-2L_val.root"]:
    # for filename in ["ttbar-2L_test.root"]: #, "ttbar-2L_train.root", "ttbar-2L_val.root"]:
        create_parquet(filename,path)
        print(f"Created {filename.replace('.root','')}_051026.parquet")


        