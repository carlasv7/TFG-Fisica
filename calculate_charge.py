import sys
import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit.Chem import GetPeriodicTable
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
INPUT     = 'archivo0001.outwfn'  # insertar aquí el archivo .outwfn
DIRINPUT  = './'                  # insertar aquí la ruta del archivo
DIROUTPUT = './'                  # insertar aquí la ruta de la salida 
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
model_paths = {
        'C': 'MODELS_definitivos/C_model.pth',
        'H': 'MODELS_definitivos/H_model.pth',
        'O': 'MODELS_definitivos/O_model.pth'
        }
#--------------------------------------------------------------------------#
architectures = {
    'C': [('Linear', 768), ('Tanh',), ('Linear', 768), ('Tanh',), ('Linear', 1)],

    'H': [('Linear', 384), ('Tanh',), ('Linear', 384), ('Tanh',), ('Linear', 384), ('Tanh',), ('Linear', 384), ('Tanh',), ('Linear', 1)],

    'O': [('Linear', 512), ('Tanh',), ('Linear', 512), ('Tanh',), ('Linear', 512), ('Tanh',), ('Linear', 1)]

}

lambdas    = {
        'C': 1e-3,
        'H': 0,
        'O': 0
        }

lrate_list = {
        'C': 1e-3,
        'H': 1e-3,
        'O': 1e-3
        }
#--------------------------------------------------------------------------#
criterion = nn.MSELoss() # loss function
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                FUNCIONES PARA EL CÁLCULO DE FEATURES                     #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
rs_tot_H  = np.array([0.85, 3.87, 13.85]) # centroide
rs_int_H  = np.array([2.36, 8.86])
eta_tot_H = np.array([0.1, 0.3, 0.6])     # anchura gaussiana
eta_int_H = np.array([0.2, 0.45])
#--------------------------------------------------------------------------#
rs_tot_O  = np.array([0.85, 3.56, 12.52]) # centroide
rs_int_O  = np.array([2.20, 8.04])
eta_tot_O = np.array([0.1, 0.3, 0.6])     # anchura gaussiana
eta_int_O = np.array([0.2, 0.45])
#--------------------------------------------------------------------------#
rs_tot_C  = np.array([1.01, 3.29, 13.17]) # centroide
rs_int_C  = np.array([2.15, 8.23])
eta_tot_C = np.array([0.1, 0.3, 0.6])     # anchura gaussiana
eta_int_C = np.array([0.2, 0.45])
#--------------------------------------------------------------------------#
def get_atomic_num(atom):
    if atom == 'H': return 1
    if atom == 'C': return 6
    if atom == 'O': return 8
#--------------------------------------------------------------------------#
def correct_charges(atoms_list):
    total_charge = sum(lista["charge"] for lista in atoms_list)

    if total_charge == 0:
        return atoms_list

    # Calcula el número de electrones asociado a cada átomo
    for lista in atoms_list:
        lista["electrons"] = get_atomic_num(lista["atom"]) - lista["charge"]
    # Calcula el porcentaje de carga asociado a cada átomo
    total_electrons = sum(lista["electrons"] for lista in atoms_list)
    for lista in atoms_list:
        lista["electrons_percent"] = lista["electrons"]/total_electrons

    # Corrige el número de electrones asociado a cada átomo
    total_corrected_electrons = total_electrons + total_charge
    for lista in atoms_list:
        lista["correct_electrons"] = lista["electrons_percent"]*total_corrected_electrons

    # Corrige la carga de cada átomo para que la suma total sea cero
    for lista in atoms_list:
        lista["correct_charge"] = get_atomic_num(lista["atom"])-lista["correct_electrons"]

    # Comprobación de que la suma total e las cargas corregidas es cero
    total_correct_charge = sum(lista["correct_charge"] for lista in atoms_list)

    correct_atoms_list = []
    for lista in atoms_list:
        correct_atoms_list.append({
            "atom": lista["atom"],
            "coords": lista["coords"],
            "charge": lista["correct_charge"],
             })
    return correct_atoms_list
#--------------------------------------------------------------------------#
def generate_graph(molecule, factor=0.5):
    bonds     = {}
    neighbors = {}

    # Obtener la tabla periódica de RDKit para radios de Van der Waals
    ptable = GetPeriodicTable()

    # Inferir enlaces por distancia y radios de van der Waals
    for i, (atom_i, r_i, charge_i) in enumerate(molecule):
        for j in range(i + 1, len(molecule)):
            atom_j, r_j, charge_j = molecule[j]
            dist = np.linalg.norm(np.array(r_i) - np.array(r_j)) 
            cutoff = factor*(ptable.GetRvdw(atom_i) + ptable.GetRvdw(atom_j))
            if dist <= cutoff:
               bonds[(i,j)] = True
               bonds[(j,i)] = True
               neighbors[i] = neighbors.get(i,[]) + [j]
               neighbors[j] = neighbors.get(j,[]) + [i]
            else:
               bonds[(i,j)] = False
               bonds[(j,i)] = False
    return bonds, neighbors
#-------------------------------------------------------------------------#
def extract_data(input_file):
    with open(input_file) as data_file: lines = data_file.readlines()
    for i, line in enumerate(lines):
        if "Title line of this file:" in line:
            smiles = line.split()[-1]
        if "Attractor" in line and "X,Y,Z coordinate (Angstrom)" in line:
            coord_i = i + 1
        if "Detecting boundary grids..." in line:
            coord_f = i # en el lines[a:b] cogerá la línea b-1
        if "Atom" in line and "Basin" in line and "Charge (e)" in line:
            charge_i = i + 1
            charge_f = charge_i + coord_f - coord_i

    coords_dict = {}
    for line in lines[coord_i:coord_f]:
        data_coord = line.split()
        basin   = data_coord[0]
        coords_dict[basin] = [float(data_coord[1]), float(data_coord[2]), float(data_coord[3])]

    charges_dict = {}
    for line in lines[charge_i:charge_f]:
        data_charge = line.split()
        basin       = data_charge[3]
        atom        = data_charge[1].strip("(")
        charge      = float(data_charge[4])
        charges_dict[basin] = {"atom"  : atom,
                               "charge": charge}

    data = {}
    for basin, coords in coords_dict.items():
        if basin in charges_dict:
            data[basin] = {"atom"  : charges_dict[basin]["atom"],
                           "charge": charges_dict[basin]["charge"],
                           "coords": coords
                           }
    atoms = []
    for basin, value in data.items():
        atoms.append({
            "atom"  : value["atom"],
            "charge": value["charge"],
            "coords": value["coords"]
            })

    correct_atoms = correct_charges(atoms)

    smiles_data = [(item["atom"], item["coords"], item["charge"]) for item in correct_atoms]

    return smiles_data, smiles
#------------------------------------------------------------------------#
def theta(coords_i, coords_j, coords_k):
    v1     = np.array(coords_j) - np.array(coords_i)
    v2     = np.array(coords_k) - np.array(coords_i)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    cos_theta = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return np.arccos(cos_theta)
#------------------------------------------------------------------------#
def phi(coords_i, coords_j, coords_k, coords_l):
    v_ij = np.array(coords_j) - np.array(coords_i)
    v_jk = np.array(coords_k) - np.array(coords_j)
    v_kl = np.array(coords_l) - np.array(coords_k)
    n_jk = np.linalg.norm(v_jk)

    v1 = np.cross(v_ij, v_jk)
    v2 = np.cross(v_jk, v_kl)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 ==0:
        return 0
    cos_phi = np.dot(v1, v2)/(n1*n2)
    sin_phi = np.dot(v_jk, (np.cross(v1, v2)))/(n_jk*n1*n2)
    return np.arctan2(sin_phi, cos_phi)
#------------------------------------------------------------------------#
def idx_lamb_s(lamb, s):
    if lamb == -1 and s == +1: return 0
    if lamb == 0 and s == -1 : return 1
    if lamb == 0 and s == +1 : return 2
    if lamb == +1 and s == -1: return 3
    return None
#------------------------------------------------------------------------#
def get_features(molecule):

    # Generar grafo a partir de coordenadas xyz
    bonds, neighbors = generate_graph(molecule)

    # Definir el diccionario de datos del smiles
    smiles_features = {
                    'atom'         : [],
                    'idx'          : [],
                    'idx_neighbors': [],
                    'features'     : [],
                    'charge'       : []
                    }

    for i, (atom_i, coords_i, charge_i) in enumerate(molecule):
        if atom_i == 'C':
            rs_tot  = rs_tot_C
            rs_int  = rs_int_C
            eta_tot = eta_tot_C
            eta_int = eta_int_C

        elif atom_i == 'H':
            rs_tot  = rs_tot_H
            rs_int  = rs_int_H
            eta_tot = eta_tot_H
            eta_int = eta_int_H

        elif atom_i == 'O':
            rs_tot  = rs_tot_O
            rs_int  = rs_int_O
            eta_tot = eta_tot_O
            eta_int = eta_int_O

        # Inicializar todos los features para cada átomo i (sobre el que se construye el sumatorio)
        f1   = len(neighbors[i])
        f2   = 0
        f3   = 0
        f4_1 = np.zeros(3)
        f4_2 = np.zeros(3)
        f5_1 = np.zeros(4)
        f5_2 = np.zeros(4)
        f6   = np.zeros((len(eta_tot), len(rs_tot)))
        f7   = np.zeros((len(eta_int), 3, len(rs_int)))
        f8   = np.zeros((len(eta_int), 4, len(rs_int)))

        for j, (atom_j, coords_j, charge_j) in enumerate(molecule):
            if j == i: continue
            Z_j  = get_atomic_num(atom_j) # número atómico
            r_ij = np.linalg.norm(np.array(coords_j) - np.array(coords_i)) # distancia entre átomos

            # Sumatorio sobre todos los vecinos de i
            if j in neighbors[i]:
               f2 += Z_j
               f3 += Z_j/r_ij

            # Sumatorio sobre todos los átomos j!=i de la molécula
            for eta_idx, eta in enumerate(eta_tot):
               for rs_idx, rs in enumerate(rs_tot):
                   f6[eta_idx, rs_idx] += Z_j/r_ij*np.exp(-eta*(r_ij - rs)**2)

            for k, (atom_k, coords_k, charge_k) in enumerate(molecule):
                if k == j: continue
                if k == i: continue

                Z_k  = get_atomic_num(atom_k)
                r_ik = np.linalg.norm(np.array(coords_k) - np.array(coords_i))

                theta_ijk = theta(coords_i, coords_j, coords_k)
                d_ijk = (r_ij + r_ik)/2

                for lamb_idx, lamb in enumerate([-1, 0, 1]):

                    # Sumatorio sobre todos los k,j!=i (k!=j) de la molécula para cada valor de eta, lambda y rs
                    for eta_idx, eta in enumerate(eta_int):
                       for rs_idx, rs in enumerate(rs_int):
                            f7[eta_idx, lamb_idx, rs_idx] += (1 + lamb*np.cos(theta_ijk) + (1 - abs(lamb))*np.sin(theta_ijk))*np.exp(-eta*(d_ijk - rs))**2*Z_j*Z_k


                    # Sumatorio sobre todos los j-vecinos de i y todos los k-vecinos de j (i,j,k distintos)
                    if j in neighbors[i] and k in neighbors[j]:
                        f4_1[lamb_idx] += (1 + lamb*np.cos(theta_ijk) + (1 - abs(lamb))*np.sin(theta_ijk))*Z_j*Z_k
                    
                    # Sumatorio sobre tods los j-vecinos de i y todos los k-vecinos de i (k,i,j distintos)
                    if j in neighbors[i] and k in neighbors[i]:
                        theta_kij = theta(coords_k, coords_i, coords_j)
                        f4_2[lamb_idx] += (1 + lamb*np.cos(theta_kij) + (1 - abs(lamb))*np.sin(theta_kij))*Z_j*Z_k

                    for l, (atom_l, coords_l, charge_l) in enumerate(molecule):
                        if l == i: continue
                        if l == j: continue
                        if l == k: continue

                        Z_l  = get_atomic_num(atom_l)
                        r_il = np.linalg.norm(np.array(coords_l) - np.array(coords_i))

                        phi_ijkl = phi(coords_i, coords_j, coords_k, coords_l)
                        
                        for s in[-1, 1]:
                            lamb_s_idx = idx_lamb_s(lamb, s)
                            if lamb_s_idx is None: continue
                            for eta_idx, eta in enumerate(eta_int):
                                for rs_idx, rs in enumerate(rs_int):
                                    d_ijkl = (r_ij + r_ik + r_il)/3
                                    f8[eta_idx, lamb_s_idx, rs_idx] += (1 + lamb*np.cos(phi_ijkl) + s*(1 - abs(lamb))*np.sin(phi_ijkl))*np.exp(-eta*(d_ijkl - rs)**2)*Z_k*Z_l*Z_j

                        # Sumatorio sobre todos los j-vecinos de i, todos los k-vecinos de j y todos los l-vecinos de k (i,j,k,l distintos)
                            if j in neighbors[i] and k in neighbors[j] and l in neighbors[k]:
                                f5_1[lamb_s_idx] += (1 + lamb*np.cos(phi_ijkl) + s*(1 - abs(lamb))*np.sin(phi_ijkl))*Z_j*Z_k*Z_l

                        # Sumatorio sobre todos los l-vecinos de i, todos los j-vecinos de i y todos los k vecinos de j (i,j,k,l distintos)
                            if l in neighbors[i] and j in neighbors[i] and k in neighbors[j]:
                                phi_lijk = phi(coords_l, coords_i, coords_j, coords_k)
                                #print('phi_lijk = ', phi_lijk*180/np.pi)
                                f5_2[lamb_s_idx] += (1 + lamb*np.cos(phi_lijk) + s*(1 - abs(lamb))*np.sin(phi_lijk))*Z_j*Z_k*Z_l


        # Almacenar resultados features
        features = []
        features.append(f1)
        features.append(f2)
        features.append(f3)
        features.extend(f4_1)
        features.extend(f4_2)
        features.extend(f5_1)
        features.extend(f5_2)
        features.extend(f6.flatten())
        features.extend(f7.flatten())
        features.extend(f8.flatten())

        smiles_features['atom'].append(atom_i)
        smiles_features['idx'].append(i)
        smiles_features['idx_neighbors'].append(neighbors[i])
        smiles_features['features'].append(features)
        smiles_features['charge'].append(charge_i)

    return smiles_features
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
#            FUNCIONES PARA LA DETERMINACIÓN DE LA CARGA                 #
#------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
def build_model(n_features, architecture):
    layers = []
    in_features = int(n_features)

    for item in architecture:
        name = item[0]

        # Capa lineal: ('Linear', out_features)
        if name == 'Linear':
            out_features = int(item[1])
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        # Dropout: ('Dropout', p)
        elif name == 'Dropout':
            p = float(item[1])
            layers.append(nn.Dropout(p))

        # Otras capas: buscar en torch.nn por nombre
        else:
            layer = getattr(nn, name)
            layers.append(layer())

    return nn.Sequential(*layers)
#------------------------------------------------------------------------#
def load_best_state_model(atom, architecture, model_path, n_features):
    model = build_model(n_features, architecture)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
# CÁLCULO DE FEATURES
smiles_data, SMILES = extract_data(os.path.join(DIRINPUT, INPUT))
smiles_features = get_features(smiles_data)

# PREDICCIÓN DE LA CARGA
loaded_models = {
    atom: load_best_state_model(atom, architectures[atom], model_paths[atom], len(smiles_features['features'][0])) for atom in ['C', 'H', 'O']}
pred_atoms = []

pred_atoms = []

for i, (atom, coords, charge) in enumerate(smiles_data):
    X = torch.tensor([smiles_features['features'][i]], dtype = torch.float32)
    model = loaded_models[atom]

    with torch.no_grad():
        pred = model(X).item()

    pred_atoms.append({
        'atom'  : atom,
        'coords': coords,
        'charge': pred
        })

corrected_pred_atoms = correct_charges(pred_atoms)

string = "%-6s %10s %10s %10s %8s %20s %12s %12s\n"  % ("Átomo", "X", "Y", "Z",  "Idx", "Vecinos", "Carga real", "Carga predicha")

for i, value in enumerate(corrected_pred_atoms):
    atom           = value['atom']
    x, y, z        = value['coords']
    corrected_pred = value['charge']
    real_charge    = smiles_features['charge'][i]
    idx            = smiles_features['idx'][i]
    neighbors      = smiles_features['idx_neighbors'][i]

    string += "%-6s %10.6f %10.6f %10.6f %8d %20s %12.6f %12.6f\n" % (atom, x, y, z, idx, str(neighbors), real_charge, corrected_pred)

os.makedirs(DIROUTPUT, exist_ok=True)

base       = os.path.basename(INPUT)       
name       = base.replace(".outwfn", "")   
outputname = 'cargas_%s.txt' % name

with open(os.path.join(DIROUTPUT, outputname), 'w') as asdf:
    asdf.write(SMILES  + '\n\n')
    asdf.write(string)
